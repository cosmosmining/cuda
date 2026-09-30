# Network Analysis for European Intercity Rail

**Team:** Jeremy Kovacs (jkovacs), Po-Chun Wu (pochunw), Johnson Tsai (chiungct)

We model Europe's intercity passenger rail as a graph (nodes = cities/stations,
edges = direct train links weighted by weekly frequency) built from GTFS
timetables, then ask:

1. How connected are city pairs, and how does that relate to distance and population?
2. Which train routes and corridors matter most?
3. Which new routes (new high-speed lines, or new direct services on existing
   track) would help most, and can a model predict where direct trains "should" exist?

**The write-up is in [`REPORT.md`](REPORT.md).** Auto-generated numbers are in
[`results/RESULTS.md`](results/RESULTS.md), figures in `results/figures/`, and
every figure's data in `results/tables/`.

> **Data status.** The committed results are computed on a **hand-compiled,
> approximate demo network** (`data/demo/lines.txt`, 310 lines / 715
> stations), because the official GTFS portals could not be reached from the
> environment this was built in. The pipeline is written for real GTFS: run
> `python -m eurail download` and `python -m eurail all` (below) to regenerate
> everything from official timetables. Treat the committed numbers as a working
> demonstration, not final findings.

## Quick start

```bash
cd eu-rail-network
pip install -r requirements.txt

# Offline, on the demo network (~4 minutes):
PYTHONPATH=src python -m eurail all --demo

# On official timetables:
PYTHONPATH=src python -m eurail download          # feeds listed in config/feeds.toml
PYTHONPATH=src python -m eurail all               # build + analyze
# or step by step:
PYTHONPATH=src python -m eurail build --week 2026-03-02
PYTHONPATH=src python -m eurail analyze --rounds 15

python -m pytest                                   # 22 tests
```

Feeds you download by hand (e.g. from the [Mobility Database](https://mobilitydatabase.org/))
only need a `[[feed]]` entry with a `path` in `config/feeds.toml`; set
`route_types` / `name_regex` to keep only intercity trains.

## Pipeline

| Step | Module | What it does |
|---|---|---|
| Ingest | `src/eurail/gtfs.py` | Reads GTFS (zip or folder), keeps rail route types, counts how often each trip runs in a reference week (`calendar` + `calendar_dates`), handles times past 24:00 and untimed stops |
| Merge stations | `src/eurail/cities.py` | Folds platforms into stations, then stations into **city nodes**: a station joins the most populous GeoNames city whose population-scaled radius (3-15 km) covers it. Airports and out-of-town high-speed stations stay separate. Populations come from the offline GeoNames copy bundled in `geonamescache` |
| Build graphs | `src/eurail/graph.py` | **Service graph**: every pair of stops on the same train is linked (TGV Paris-Lyon-Marseille links all 3 pairs), so path length = number of trains. **Segment graph**: consecutive stops only (the physical corridors). Edges carry trains/week and weekly-weighted median travel time. Cross-border trains appearing in two feeds are counted once |
| Metrics | `src/eurail/metrics.py` | Structure, centrality, **generalized travel time** (ride + half the headway + 15 min per change), gravity demand, demand-weighted accessibility, flow assignment |
| RQ1 | `src/eurail/gravity.py` | PPML / logit / OLS gravity models with a same-country (border) effect; under-served pairs |
| RQ2 | `src/eurail/importance.py` | Knock-out experiments: cancel each route, close each track segment, measure the increase in mean journey time |
| RQ3 | `src/eurail/recommend.py` | Greedy budgeted network design with an exact incremental shortest-path update; two scenarios (new high-speed line vs new direct service) |
| RQ3 (ML) | `src/eurail/linkpred.py` | Gradient boosting / logistic regression link prediction with leakage-free cross-validation; "missing links" |
| Output | `src/eurail/analyze.py`, `plotting.py` | Tables, figures, `summary.json`, `RESULTS.md` |

## Repository layout

```
config/feeds.toml            official GTFS sources + per-feed filters
data/demo/                   hand-compiled demo network (lines.txt, stations.csv) + generated GTFS zip
data/reference/              Natural Earth borders (map background, sea-crossing check)
data/processed/              built network: nodes, service/segment edges, trips
results/                     figures, tables, summary.json, RESULTS.md
scripts/make_borders.py      regenerates the borders file from the world-atlas npm package
src/eurail/                  the package (see table above)
tests/                       pytest suite
```

## Data sources and licences

* Timetables: GTFS feeds listed in `config/feeds.toml` (licences per publisher).
* City populations/coordinates: [GeoNames](https://www.geonames.org/) (CC BY 4.0) via `geonamescache`.
* Borders: [Natural Earth](https://www.naturalearthdata.com/) 1:50m (public domain) via
  [world-atlas](https://github.com/topojson/world-atlas) (ISC).
