"""Static figures for the report (matplotlib).

Palette and mark specs follow a validated categorical order (blue, orange,
aqua, yellow), a single-hue blue sequential ramp, recessive hairline grids,
2 px lines and direct labels on the few marks the story is about. Every figure
has a matching CSV table in results/tables for exact values.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection, PatchCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LAND = "#f0efec"
BORDER = "#d9d8d1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
EXTENT = (-11.0, 31.5, 35.5, 67.5)  # lon_min, lon_max, lat_min, lat_max


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _basemap(ax, borders_path: Path) -> None:
    feats = json.loads(Path(borders_path).read_text())["features"]
    patches = [Polygon(np.asarray(poly[0]), closed=True) for f in feats for poly in f["geometry"]["coordinates"]]
    ax.add_collection(PatchCollection(patches, facecolor=LAND, edgecolor=BORDER, linewidth=0.4, zorder=0))
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect(1 / np.cos(np.radians(50)))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _xy(nodes: pd.DataFrame) -> dict:
    return {k: (lo, la) for k, lo, la in zip(nodes.node_id, nodes.lon, nodes.lat)}


_HALO = [matplotlib.patheffects.withStroke(linewidth=2.2, foreground=SURFACE)]


class _Placer:
    """Greedy collision-free placement of labels/badges (in display space)."""

    def __init__(self, ax):
        self.ax = ax
        ax.figure.canvas.draw()
        self.renderer = ax.figure.canvas.get_renderer()
        self.boxes = []

    def _free(self, artist) -> bool:
        bb = artist.get_window_extent(self.renderer).expanded(1.04, 1.08)
        if any(bb.overlaps(b) for b in self.boxes):
            return False
        self.boxes.append(bb)
        return True

    def label(self, x, y, text, size=7.5) -> bool:
        for ha, va, dx, dy in (("left", "bottom", 3, 2), ("right", "bottom", -3, 2), ("left", "top", 3, -2), ("right", "top", -3, -2)):
            t = self.ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=size,
                                 color=INK, zorder=6, path_effects=_HALO)
            if self._free(t):
                return True
            t.remove()
        return False

    def badge(self, x1, y1, x2, y2, text, color) -> None:
        for f in (0.5, 0.35, 0.65, 0.2, 0.8):
            t = self.ax.text(x1 + f * (x2 - x1), y1 + f * (y2 - y1), text, fontsize=7, color=SURFACE, ha="center",
                             va="center", zorder=7, weight="bold",
                             bbox=dict(boxstyle="circle,pad=0.18", facecolor=color, edgecolor=SURFACE, linewidth=1))
            if self._free(t) or f == 0.8:
                return
            t.remove()



def network_map(nodes, segment, centrality, borders, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8, 9.4))
    _basemap(ax, borders)
    xy = _xy(nodes)
    seg = segment.sort_values("trains_per_week")
    lines = [[xy[a], xy[b]] for a, b in zip(seg.u, seg.v)]
    w = seg.trains_per_week.to_numpy()
    norm = np.log10(np.clip(w, 7, None)) / np.log10(max(w.max(), 8))
    ax.add_collection(LineCollection(lines, colors=BLUES(0.25 + 0.75 * norm), linewidths=0.4 + 2.6 * norm,
                                     capstyle="round", zorder=2))
    c = centrality.set_index("node_id").reindex(nodes.node_id)
    s = c.strength.fillna(0).to_numpy()
    ax.scatter(nodes.lon, nodes.lat, s=3 + 60 * np.sqrt(s / s.max()), color=BLUE_RAMP[5], edgecolor=SURFACE,
               linewidth=0.6, zorder=3)
    placer = _Placer(ax)
    for r in c.sort_values("strength", ascending=False).head(40).itertuples():
        placer.label(r.lon, r.lat, r.name)
    ax.set_title("European intercity rail network: corridors by weekly trains")
    ax.text(0, -0.02, "Line width and shade: trains per week on each corridor (both directions, log scale). "
            "Dot size: node strength (trains/week over all direct links).", transform=ax.transAxes,
            fontsize=7.5, color=INK2, va="top")
    _save(fig, path)


def bars(df: pd.DataFrame, label_col: str, value_col: str, title: str, xlabel: str, path: Path,
         fmt: str = "{:.2f}", color: str = SERIES[0], note: str | None = None) -> None:
    _style()
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(d) + 1.2))
    ax.barh(d[label_col], d[value_col], height=0.55, color=color, zorder=2)
    for yv, v in enumerate(d[value_col]):
        ax.text(v, yv, "  " + fmt.format(v), va="center", fontsize=7.5, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, d[value_col].max() * 1.18)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title)
    if note:
        fig.text(0.01, -0.01, note, fontsize=7.5, color=INK2, va="top")
    _save(fig, path)


def two_bar_panels(left: pd.DataFrame, right: pd.DataFrame, cols: tuple, titles: tuple, xlabels: tuple, path: Path) -> None:
    _style()
    n = max(len(left), len(right))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.3 * n + 1.3))
    for ax, df, (lab, val), t, xl in zip(axes, (left, right), cols, titles, xlabels):
        d = df.iloc[::-1]
        ax.barh(d[lab], d[val], height=0.55, color=SERIES[0], zorder=2)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(t)
        ax.set_xlabel(xl)
    fig.tight_layout()
    _save(fig, path)


def degree_ccdf(service_deg: np.ndarray, segment_deg: np.ndarray, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for deg, lab, col in ((service_deg, "Service graph (direct-train links)", SERIES[0]),
                          (segment_deg, "Segment graph (consecutive stops)", SERIES[1])):
        k = np.sort(deg[deg > 0])
        ccdf = 1.0 - np.arange(len(k)) / len(k)
        ax.step(k, ccdf, where="post", color=col, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree k")
    ax.set_ylabel("Share of nodes with degree >= k")
    ax.set_title("Degree distribution (complementary CDF)")
    ax.legend(loc="lower left")
    _save(fig, path)


def gravity_panels(pairs: pd.DataFrame, coefs: pd.DataFrame, path: Path) -> None:
    """Left: observed mean direct trains (zeros included) per gravity-index bin vs the PPML fit.
    Right: generalized effective speed vs distance."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    d = pairs[(pairs.dist_km >= 50) & (pairs.dist_km <= 1500)].copy()
    pp = coefs[coefs.model == "ppml_direct_trains"].set_index("term").coef
    d["fit"] = np.exp(pp["Intercept"] + pp["log_pop_prod"] * d.log_pop_prod + pp["log_dist"] * d.log_dist
                      + pp["same_country"] * d.same_country)
    d["g"] = d.log_pop_prod - 2 * d.log_dist
    edges = np.quantile(d.g, np.linspace(0, 1, 15))
    d["bin"] = np.clip(np.searchsorted(edges, d.g, side="right") - 1, 0, len(edges) - 2)
    ax = axes[0]
    for flag, lab, col in ((1, "Same country", SERIES[0]), (0, "Cross-border", SERIES[1])):
        b = d[d.same_country == flag].groupby("bin").agg(x=("g", "mean"), obs=("trains_per_week", "mean"),
                                                        fit=("fit", "mean"), n=("g", "size"))
        b = b[(b.n >= 20) & (b.obs > 0)]
        ax.plot(np.exp(b.x), b.fit, color=col, linewidth=2, label=f"{lab}: PPML fit")
        ax.scatter(np.exp(b.x), b.obs, s=34, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=f"{lab}: observed mean")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gravity index  P_i P_j / d_ij^2 (bin mean)")
    ax.set_ylabel("Mean direct trains per week (incl. pairs with none)")
    ax.set_title("Direct service vs city size and distance")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    r = pairs[np.isfinite(pairs.journey_min) & (pairs.dist_km >= 50) & (pairs.dist_km <= 1500)]
    ax.scatter(r.dist_km, r.eff_speed_kmh, s=3, color=BLUE_RAMP[2], alpha=0.25, edgecolor="none", zorder=1)
    bins = np.arange(50, 1550, 100)
    mid = (bins[:-1] + bins[1:]) / 2
    med = [r.eff_speed_kmh[(r.dist_km >= a) & (r.dist_km < b)].median() for a, b in zip(bins[:-1], bins[1:])]
    ax.plot(mid, med, color=BLUE_RAMP[6], linewidth=2, zorder=3)
    ax.text(mid[-1], med[-1], "  median", color=INK2, fontsize=8, va="center")
    ax.set_xlabel("Straight-line distance (km)")
    ax.set_ylabel("Effective speed (km/h, generalized)")
    ax.set_title("Door-to-door rail speed rises with distance")
    fig.tight_layout()
    _save(fig, path)


def highlight_map(nodes, segment, links: pd.DataFrame, borders, path: Path, title: str, note: str,
                  groups: list[tuple[str, str, str]] | None = None, badges: bool = True, width: float = 2.2) -> None:
    """``links`` columns: a, b (node ids), rank, group. ``groups``: (group, legend label, colour)."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 9.4))
    _basemap(ax, borders)
    xy = _xy(nodes)
    ax.add_collection(LineCollection([[xy[a], xy[b]] for a, b in zip(segment.u, segment.v)], colors=AXIS,
                                     linewidths=0.6, zorder=1))
    groups = groups or [("default", None, SERIES[1])]
    names = nodes.set_index("node_id")["name"]
    drawn = []
    for g, lab, col in groups:
        sub = links[links.group == g] if "group" in links else links
        styles = {"linestyle": (0, (4, 2))} if g == "direct" else {}
        for r in sub.itertuples():
            (x1, y1), (x2, y2) = xy[r.a], xy[r.b]
            ax.plot([x1, x2], [y1, y2], color=col, linewidth=width, zorder=3, **styles)
            ax.scatter([x1, x2], [y1, y2], s=22, color=col, edgecolor=SURFACE, linewidth=1.2, zorder=4)
            drawn.append((x1, y1, x2, y2, str(r.rank), col, r.a, r.b))
        if lab:
            ax.plot([], [], color=col, linewidth=2.2, label=lab, **styles)
    placer = _Placer(ax)
    for x1, y1, x2, y2, rank, col, *_ in drawn if badges else []:
        placer.badge(x1, y1, x2, y2, rank, col)
    labelled = set()
    for *_, a, b in drawn:
        for n_ in (a, b):
            if n_ not in labelled:
                labelled.add(n_)
                placer.label(*xy[n_], names[n_], size=7)
    if any(lab for _, lab, _ in groups):
        ax.legend(loc="upper left")
    ax.set_title(title)
    ax.text(0, -0.02, note, transform=ax.transAxes, fontsize=7.5, color=INK2, va="top", wrap=True)
    _save(fig, path)


def accessibility_map(centrality: pd.DataFrame, segment, nodes, borders, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.8, 9.4))
    _basemap(ax, borders)
    xy = _xy(nodes)
    ax.add_collection(LineCollection([[xy[a], xy[b]] for a, b in zip(segment.u, segment.v)], colors=AXIS,
                                     linewidths=0.5, zorder=1))
    c = centrality.dropna(subset=["accessibility_min"])
    lo, hi = np.nanpercentile(c.accessibility_min, [2, 98])
    sc = ax.scatter(c.lon, c.lat, c=c.accessibility_min.clip(lo, hi), cmap=BLUES, s=8 + 40 * np.sqrt(c.population / c.population.max()),
                    edgecolor=SURFACE, linewidth=0.6, zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.01)
    cb.set_label("Demand-weighted mean journey time to other cities (min)", color=INK2)
    cb.outline.set_visible(False)
    placer = _Placer(ax)
    for r in pd.concat([c.nlargest(12, "population"), c.nsmallest(6, "accessibility_min"),
                        c.nlargest(6, "accessibility_min")]).drop_duplicates("node_id").itertuples():
        placer.label(r.lon, r.lat, r.name, size=7)
    ax.set_title("Accessibility: how quickly each city reaches the rest of Europe by rail")
    ax.text(0, -0.02, "Darker = longer mean generalized journey (in-vehicle + wait + 15 min per change, capped by a road "
            "alternative), weighted by gravity demand. Dot size: population.", transform=ax.transAxes, fontsize=7.5,
            color=INK2, va="top")
    _save(fig, path)


def greedy_curve(picks: pd.DataFrame, path: Path, xlabel: str, title: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.concatenate([[0], picks.cum_cost])
    y = np.concatenate([[0], picks.cum_gain_min])
    ax.plot(x, y, color=SERIES[0], marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    for r in picks.head(6).itertuples():
        ax.text(r.cum_cost, r.cum_gain_min, f"  {r.round}. {r.city_a}-{r.city_b}", fontsize=7.5, color=INK2, va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative reduction in mean journey time (min)")
    ax.set_title(title)
    ax.set_ylim(0, y.max() * 1.15)
    _save(fig, path)


def roc_curves(curves: dict, scores: pd.DataFrame, keys: list[str], path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.8, 5))
    auc = scores.set_index("model").roc_auc
    pretty = {"gradient_boosting (full)": "Gradient boosting, all features",
              "gradient_boosting (fundamentals)": "Gradient boosting, fundamentals only",
              "gravity_baseline": "Gravity score alone", "adamic_adar_baseline": "Adamic-Adar alone"}
    for k, col in zip(keys, SERIES):
        fpr, tpr, _ = curves[k]
        ax.plot(fpr, tpr, color=col, label=f"{pretty.get(k, k)}  (AUC {auc[k]:.3f})")
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Link prediction: does a pair have a daily direct train?")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    _save(fig, path)
