"""Run every analysis on a built network and write tables, figures and a summary."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import gravity, importance, linkpred, metrics, plotting, recommend
from .geo import LandMask
from .graph import RailNetwork


def _fmt_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v) if np.isfinite(v) else "-")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def run(net: RailNetwork, outdir: str | Path, borders: str | Path, meta: dict | None = None,
        greedy_rounds: int = 10, log=print) -> dict:
    out = Path(outdir)
    tables, figs = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary: dict = {"build": meta or {}}

    # ---- structure & centrality -----------------------------------------------------------
    model = metrics.build_cost_model(net.nodes, net.service_edges)
    costs = metrics.edge_cost(net.service_edges.time_median, net.service_edges.trains_per_week)
    R = model.od_matrix(costs)
    summary["service_graph"] = metrics.network_summary(net.service_graph())
    summary["segment_graph"] = metrics.network_summary(net.segment_graph())
    summary["accessibility"] = {
        "od_zones": int(len(model.od)),
        "mean_generalized_journey_min": model.mean_time(R),
        "rail_competitive_demand_share": model.rail_share(R),
        "demand_share_within_3h": model.share_within(R, 180),
        "demand_share_within_5h": model.share_within(R, 300),
    }
    log(f"[{time.time() - t0:5.1f}s] structure done: {summary['service_graph']['nodes']} nodes")
    cen = metrics.centrality_table(net.nodes, net.service_edges, net.segment_edges, model, costs)
    cen.to_csv(tables / "node_centrality.csv", index=False, float_format="%.5g")
    _, edge_load = metrics.flow_assignment(model, costs)
    names = net.nodes.set_index("node_id")["name"]
    loads = net.service_edges.assign(demand_load=edge_load, a=net.service_edges.u.map(names), b=net.service_edges.v.map(names))
    loads.sort_values("demand_load", ascending=False).to_csv(tables / "service_edge_loads.csv", index=False, float_format="%.5g")
    log(f"[{time.time() - t0:5.1f}s] centrality done")

    # ---- research question 1: connectivity vs distance & population ------------------------
    pairs = gravity.od_pair_table(net.nodes, net.service_edges, model, R)
    grav = gravity.fit_gravity_models(pairs)
    grav["coefficients"].to_csv(tables / "gravity_coefficients.csv", index=False, float_format="%.5g")
    under = gravity.underserved_pairs(grav["residuals"])
    under.to_csv(tables / "underserved_pairs.csv", index=False, float_format="%.4g")
    summary["gravity"] = grav["stats"]
    log(f"[{time.time() - t0:5.1f}s] gravity done")

    # ---- research question 2: most important routes and corridors -------------------------
    agg = importance.ServiceAggregator(net)
    routes = importance.route_importance(net, model, agg)
    routes.to_csv(tables / "route_importance.csv", index=False, float_format="%.5g")
    corridors = importance.corridor_importance(net, model, agg)
    corridors.to_csv(tables / "corridor_importance.csv", index=False, float_format="%.5g")
    log(f"[{time.time() - t0:5.1f}s] importance done")

    # ---- research question 3: which new routes? --------------------------------------------
    land = LandMask(borders)
    hsr = recommend.hsr_candidates(net.nodes, model, R, land=land)
    hsr_scored = recommend.label(recommend.score_candidates(model, R, hsr), net.nodes, model)
    hsr_scored.to_csv(tables / "hsr_candidates_scored.csv", index=False, float_format="%.4g")
    hsr_pick = recommend.label(recommend.greedy_select(model, R, hsr, k=greedy_rounds), net.nodes, model)
    hsr_pick.to_csv(tables / "hsr_greedy_plan.csv", index=False, float_format="%.4g")
    direct = recommend.direct_service_candidates(net, model, R)
    direct_pick = recommend.label(recommend.greedy_select(model, R, direct, k=greedy_rounds), net.nodes, model)
    direct_pick.to_csv(tables / "direct_service_greedy_plan.csv", index=False, float_format="%.4g")
    summary["recommendations"] = {
        "hsr_candidates": int(len(hsr)), "direct_candidates": int(len(direct)),
        "hsr_plan_cost_bn_eur": float(hsr_pick.cum_cost.iloc[-1]) if len(hsr_pick) else 0.0,
        "hsr_plan_gain_min": float(hsr_pick.cum_gain_min.iloc[-1]) if len(hsr_pick) else 0.0,
        "direct_plan_gain_min": float(direct_pick.cum_gain_min.iloc[-1]) if len(direct_pick) else 0.0,
    }
    log(f"[{time.time() - t0:5.1f}s] recommendations done")

    lp = linkpred.evaluate(net, model, pairs)
    lp["scores"].to_csv(tables / "linkpred_scores.csv", index=False, float_format="%.4f")
    miss_full, coef_full = linkpred.missing_links(net, model, pairs, features="full")
    miss_fund, coef_fund = linkpred.missing_links(net, model, pairs, features="fundamentals")
    miss_full.to_csv(tables / "missing_links_full_model.csv", index=False, float_format="%.4g")
    miss_fund.to_csv(tables / "missing_links_fundamentals_model.csv", index=False, float_format="%.4g")
    pd.DataFrame({"full": coef_full, "fundamentals": coef_fund}).to_csv(tables / "linkpred_logit_coefficients.csv", float_format="%.4f")
    summary["link_prediction"] = {"pairs": lp["pairs"], "positives": lp["positives"],
                                  "scores": lp["scores"].to_dict("records")}
    log(f"[{time.time() - t0:5.1f}s] link prediction done")

    # ---- figures ------------------------------------------------------------------------------
    nodes, seg = net.nodes, net.segment_edges
    plotting.network_map(nodes, seg, cen, borders, figs / "network_map.png")
    plotting.accessibility_map(cen, seg, nodes, borders, figs / "accessibility_map.png")
    Gs, Gt = net.service_graph(), net.segment_graph()
    plotting.degree_ccdf(np.array([d for _, d in Gs.degree()]), np.array([d for _, d in Gt.degree()]),
                         figs / "degree_distribution.png")
    top_b = cen.head(15)
    top_f = cen.sort_values("transfer_flow", ascending=False).head(15)
    plotting.two_bar_panels(top_b, top_f.assign(transfer_flow=100 * top_f.transfer_flow),
                            (("name", "betweenness"), ("name", "transfer_flow")),
                            ("Betweenness (generalized-time paths)", "Demand passing through (% of OD demand)"),
                            ("Normalised betweenness", "% of demand"), figs / "hubs.png")
    plotting.gravity_panels(pairs, grav["coefficients"], figs / "gravity.png")
    rt = routes.head(15).assign(label=lambda d: d.origin + " - " + d.destination + " (" + d.route_name + ")")
    plotting.bars(rt, "label", "delta_mean_min", "Most important routes: cancel the route, how much slower is travel?",
                  "Increase in demand-weighted mean journey time (min)", figs / "route_importance.png")
    top_c = corridors.head(15)
    plotting.highlight_map(nodes, seg, pd.DataFrame({"a": top_c.u, "b": top_c.v, "rank": range(1, len(top_c) + 1),
                                                     "group": "default"}), borders,
                           figs / "critical_corridors_map.png", "Critical corridors: the 15 segments whose closure hurts most",
                           "Orange: the 15 segments with the largest increase in demand-weighted mean journey time when closed "
                           "(all trains over them cut); ranking in critical_corridors.png. Grey: rest of the network.",
                           badges=False, width=4.0)
    plotting.bars(top_c, "segment", "delta_mean_min", "Critical corridors: close the segment, how much slower is travel?",
                  "Increase in demand-weighted mean journey time (min)", figs / "critical_corridors.png")
    rec_links = pd.concat([
        pd.DataFrame({"a": model.node_ids[model.od[hsr_pick.p.astype(int)]], "b": model.node_ids[model.od[hsr_pick.q.astype(int)]],
                      "rank": hsr_pick["round"], "group": "hsr"}),
        pd.DataFrame({"a": model.node_ids[model.od[direct_pick.p.astype(int)]], "b": model.node_ids[model.od[direct_pick.q.astype(int)]],
                      "rank": direct_pick["round"], "group": "direct"}),
    ]) if len(hsr_pick) and len(direct_pick) else pd.DataFrame(columns=["a", "b", "rank", "group"])
    plotting.highlight_map(nodes, seg, rec_links, borders, figs / "recommendations_map.png",
                           "Recommended additions (greedy, best benefit per cost first)",
                           "Numbers give the greedy pick order within each scenario; links drawn as straight lines. New high-speed "
                           "lines: 200 km/h average, 16 trains/day each way. New direct services: fastest existing track, 8/day each way.",
                           groups=[("hsr", "New high-speed line", plotting.SERIES[0]),
                                   ("direct", "New direct service on existing track", plotting.SERIES[1])])
    if len(hsr_pick):
        plotting.greedy_curve(hsr_pick, figs / "hsr_greedy_curve.png", "Cumulative construction cost (bn EUR)",
                              "Greedy high-speed plan: benefit vs cost")
    keys = ["gradient_boosting (full)", "gradient_boosting (fundamentals)", "gravity_baseline", "adamic_adar_baseline"]
    plotting.roc_curves(lp["curves"], lp["scores"], keys, figs / "linkpred_roc.png")
    log(f"[{time.time() - t0:5.1f}s] figures done")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_results_md(out, summary, cen, routes, corridors, grav, under, hsr_pick, direct_pick, lp, miss_fund, miss_full, loads)
    log(f"[{time.time() - t0:5.1f}s] wrote {out}")
    return summary


def write_results_md(out: Path, s: dict, cen, routes, corridors, grav, under, hsr_pick, direct_pick, lp,
                     miss_fund, miss_full, loads) -> None:
    sg, tg, acc, gs = s["service_graph"], s["segment_graph"], s["accessibility"], s["gravity"]
    feeds = ", ".join(f"{f['name']} (week of {f['reference_week']})" for f in s["build"].get("feeds", []))
    L = [
        "# Results (auto-generated)",
        "",
        "Generated by `python -m eurail analyze`. Every number below is recomputed from the network in",
        f"`data/processed/`; built from: {feeds or 'n/a'}.",
        "",
        "## 1. Network structure",
        "",
        "| Metric | Service graph (direct trains) | Segment graph (consecutive stops) |",
        "|---|---|---|",
    ]
    for k, lab, f in [("nodes", "Nodes", "{:,}"), ("edges", "Edges", "{:,}"), ("density", "Density", "{:.4f}"),
                      ("components", "Connected components", "{}"), ("largest_component_share", "Largest component share", "{:.1%}"),
                      ("mean_degree", "Mean degree", "{:.2f}"), ("max_degree", "Max degree", "{}"),
                      ("avg_clustering", "Average clustering", "{:.3f}"), ("degree_assortativity", "Degree assortativity", "{:.3f}"),
                      ("lcc_diameter_hops", "Diameter (hops, largest component)", "{}"),
                      ("lcc_mean_hops", "Mean shortest path (hops)", "{:.2f}")]:
        L.append(f"| {lab} | {f.format(sg[k])} | {f.format(tg[k])} |")
    L += [
        "",
        f"On the service graph a path of *k* hops means *k* trains, so a typical pair of stations in the main "
        f"component is {sg['lcc_mean_hops']:.2f} trains ({sg['lcc_mean_hops'] - 1:.2f} changes) apart.",
        "",
        f"Accessibility over {acc['od_zones']} OD zones (cities >= 100k): demand-weighted mean generalized journey "
        f"**{acc['mean_generalized_journey_min']:.0f} min**; rail beats the road alternative for "
        f"**{acc['rail_competitive_demand_share']:.0%}** of demand; {acc['demand_share_within_3h']:.0%} of demand is within "
        f"3 h and {acc['demand_share_within_5h']:.0%} within 5 h by rail.",
        "",
        "### Top hubs",
        "",
        _fmt_table(cen.head(15)[["name", "country", "degree", "strength", "betweenness", "transfer_flow", "accessibility_min"]], "{:.3f}"),
        "",
        "### Busiest direct links by assigned demand",
        "",
        _fmt_table(loads.sort_values("demand_load", ascending=False).head(12)[["a", "b", "trains_per_week", "time_median", "demand_load"]], "{:.3f}"),
        "",
        "## 2. Connectivity vs distance and population (gravity models)",
        "",
        f"{gs['pairs_used']:,} city pairs 50-1500 km apart, {gs['pairs_with_direct_link']:,} with a daily direct train.",
        "",
        f"* Direct trains scale with (population product)^{gs['size_elasticity_trains']:.2f} and distance^{gs['distance_elasticity_trains']:.2f} (PPML).",
        f"* **Border effect:** a domestic pair gets **{gs['border_effect_trains']:.1f}x** the direct trains of an otherwise identical "
        f"cross-border pair (odds of any daily direct link x{gs['border_effect_link_odds']:.1f}).",
        f"* Journey time grows with distance^{gs['distance_elasticity_time']:.2f}: doubling the distance multiplies the time by "
        f"{2 ** gs['distance_elasticity_time']:.2f}, i.e. long trips are relatively faster (high-speed lines, fewer changes).",
        f"* Fit: PPML pseudo-R2 {gs['ppml_pseudo_r2']:.2f}, logit pseudo-R2 {gs['logit_pseudo_r2']:.2f}, OLS R2 {gs['ols_r2']:.2f}.",
        "",
        _fmt_table(grav["coefficients"][["model", "term", "coef", "std_err", "p_value"]], "{:.3g}"),
        "",
        "### Most under-served pairs (slower than the model expects, weighted by demand)",
        "",
        _fmt_table(under.head(12), "{:.0f}"),
        "",
        "## 3. Most important routes and corridors (knock-out experiments)",
        "",
        _fmt_table(routes.head(15)[["route_name", "origin", "destination", "trips_per_week", "delta_mean_min", "delta_mean_pct", "rail_share_lost"]], "{:.3f}"),
        "",
        _fmt_table(corridors.head(15)[["segment", "trains_per_week", "dist_km", "delta_mean_min", "demand_slower", "rail_share_lost"]], "{:.3f}"),
        "",
        "## 4. Recommended new routes",
        "",
        "### Greedy high-speed plan (cost = bn EUR construction)",
        "",
        _fmt_table(hsr_pick[["round", "city_a", "city_b", "countries", "dist_km", "sea_km", "ride_min", "cost", "gain_min", "cum_cost", "cum_gain_min"]], "{:.2f}")
        if len(hsr_pick) else "_no candidates_",
        "",
        "### Greedy new direct services on existing track (cost = thousand train-km per day)",
        "",
        _fmt_table(direct_pick[["round", "city_a", "city_b", "countries", "dist_km", "ride_min", "cost", "gain_min", "cum_gain_min"]], "{:.2f}")
        if len(direct_pick) else "_no candidates_",
        "",
        "### Link prediction (5-fold CV)",
        "",
        _fmt_table(lp["scores"], "{:.3f}"),
        "",
        "Top 'missing links' - fundamentals model (size, distance, border, track time):",
        "",
        _fmt_table(miss_fund.head(15), "{:.2f}"),
        "",
        "Top 'missing links' - full model (adds graph topology):",
        "",
        _fmt_table(miss_full.head(15), "{:.2f}"),
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(L))
