"""Command line: python -m eurail {demo-feed,download,build,analyze,all}."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO_FEED = ROOT / "data/demo/eu_intercity_demo_gtfs.zip"
BORDERS = ROOT / "data/reference/europe_borders.geojson"


def cmd_demo_feed(args) -> None:
    from .demo import generate_demo_feed

    info = generate_demo_feed(DEMO_FEED, ROOT / "data/demo/lines.txt", ROOT / "data/demo/stations.csv")
    print(f"demo feed: {info['lines']} lines, {info['stations']} stations, {info['trips']} trips -> {DEMO_FEED}")


def cmd_download(args) -> None:
    from .build import load_feed_specs

    for spec in load_feed_specs(args.config, ROOT):
        if not spec.enabled or not spec.url:
            continue
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                print(f"downloading {spec.name}: {spec.url}")
                req = urllib.request.Request(spec.url, headers={"User-Agent": "eurail-network-analysis"})
                with urllib.request.urlopen(req, timeout=120) as r, open(spec.path, "wb") as fh:
                    fh.write(r.read())
                break
            except Exception as e:  # network errors: retry with backoff, then report
                print(f"  failed ({e}); retry {attempt + 1}/4", file=sys.stderr)
                time.sleep(2 ** (attempt + 1))
        else:
            print(f"  giving up on {spec.name}; download it manually to {spec.path}", file=sys.stderr)


def cmd_build(args) -> None:
    from .build import FeedSpec, build_from_feeds, load_feed_specs, save

    if args.demo:
        if not DEMO_FEED.exists():
            cmd_demo_feed(args)
        specs = [FeedSpec("demo", DEMO_FEED)]
    else:
        specs = [s for s in load_feed_specs(args.config, ROOT) if s.enabled and s.path.exists()]
        missing = [s.name for s in load_feed_specs(args.config, ROOT) if s.enabled and not s.path.exists()]
        if missing:
            print(f"skipping feeds not downloaded yet: {missing}", file=sys.stderr)
    week = date.fromisoformat(args.week) if args.week else None
    res = build_from_feeds(specs, week=week, min_city_population=args.min_city_pop)
    save(res, args.out)
    print(json.dumps(res.meta, indent=2))


def cmd_analyze(args) -> None:
    from .analyze import run
    from .build import load

    net = load(args.processed)
    meta_path = Path(args.processed) / "build_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    run(net, args.results, BORDERS, meta=meta, greedy_rounds=args.rounds)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="eurail", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo-feed", help="generate the offline demo GTFS feed")
    d = sub.add_parser("download", help="download the official GTFS feeds in the config")
    d.add_argument("--config", default=str(ROOT / "config/feeds.toml"))
    for name in ("build", "all"):
        b = sub.add_parser(name, help="build the network" if name == "build" else "build + analyze")
        b.add_argument("--demo", action="store_true", help="use the offline demo feed")
        b.add_argument("--config", default=str(ROOT / "config/feeds.toml"))
        b.add_argument("--week", help="reference week start (Monday, YYYY-MM-DD); default: busiest week per feed")
        b.add_argument("--min-city-pop", type=int, default=15000)
        b.add_argument("--out", "--processed", dest="out", default=str(ROOT / "data/processed"))
        if name == "all":
            b.add_argument("--results", default=str(ROOT / "results"))
            b.add_argument("--rounds", type=int, default=10)
    a = sub.add_parser("analyze", help="run all analyses on a built network")
    a.add_argument("--processed", default=str(ROOT / "data/processed"))
    a.add_argument("--results", default=str(ROOT / "results"))
    a.add_argument("--rounds", type=int, default=10, help="greedy picks per scenario")
    args = p.parse_args(argv)
    if args.cmd == "demo-feed":
        cmd_demo_feed(args)
    elif args.cmd == "download":
        cmd_download(args)
    elif args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)
    elif args.cmd == "all":
        cmd_build(args)
        args.processed = args.out
        cmd_analyze(args)


if __name__ == "__main__":
    main()
