"""Supervised link prediction: which city pairs "should" have a direct train?

Target: an OD-zone pair has a direct service at least daily (>= 7 trains/week).
Features (per pair):

* exogenous - log distance, log population product, |log population ratio|,
  same country;
* topological, computed on the service graph *with the test links removed* -
  common neighbours, Jaccard, Adamic-Adar, resource allocation, log degree
  product;
* infrastructure - log fastest track time on the segment graph (is there a
  corridor, and how fast is it?).

Two feature sets are compared: ``full`` (all of the above) and
``fundamentals`` (exogenous + infrastructure only). The service graph is a
union of cliques (one per train), so topology alone predicts links very well;
the fundamentals model instead asks "given the size of the cities, their
distance and the track between them, should there be a direct train?".

Models are evaluated with stratified 5-fold cross-validation (ROC AUC and
average precision) against two single-score baselines (gravity score and
Adamic-Adar). Pairs the fitted model rates as likely but that have no direct
service are "missing links" - candidates for new routes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .graph import RailNetwork
from .metrics import CostModel

FEATURES = ["log_dist", "log_pop_prod", "abs_log_pop_ratio", "same_country",
            "common_neighbors", "jaccard", "adamic_adar", "resource_alloc", "log_degree_prod", "log_track_min"]
FUNDAMENTALS = ["log_dist", "log_pop_prod", "abs_log_pop_ratio", "same_country", "log_track_min"]
FEATURE_SETS = {"full": FEATURES, "fundamentals": FUNDAMENTALS}


def _neighbor_sets(edges_u: np.ndarray, edges_v: np.ndarray, n: int) -> list[set]:
    nb = [set() for _ in range(n)]
    for a, b in zip(edges_u, edges_v):
        nb[a].add(b)
        nb[b].add(a)
    return nb


def pair_features(pairs: pd.DataFrame, nb: list[set], track_min: np.ndarray) -> pd.DataFrame:
    """``pairs`` has integer node indices ``ia``/``ib``; neighbours of each other are ignored."""
    deg = np.array([len(s) for s in nb], dtype=float)
    cn, jac, aa, ra, dp = [], [], [], [], []
    for a, b in zip(pairs.ia, pairs.ib):
        na, nbb = nb[a] - {b}, nb[b] - {a}
        common = na & nbb
        union = na | nbb
        cn.append(len(common))
        jac.append(len(common) / len(union) if union else 0.0)
        aa.append(sum(1.0 / np.log(deg[z]) for z in common if deg[z] > 1))
        ra.append(sum(1.0 / deg[z] for z in common))
        dp.append(np.log1p(len(na)) + np.log1p(len(nbb)))
    out = pd.DataFrame({
        "log_dist": np.log(pairs.dist_km),
        "log_pop_prod": np.log(pairs.pop_a.astype(float)) + np.log(pairs.pop_b.astype(float)),
        "abs_log_pop_ratio": np.abs(np.log(pairs.pop_a.astype(float) / pairs.pop_b.astype(float))),
        "same_country": pairs.same_country.astype(int),
        "common_neighbors": cn, "jaccard": jac, "adamic_adar": aa, "resource_alloc": ra, "log_degree_prod": dp,
        "log_track_min": np.log(np.minimum(track_min, 3000.0)),
    }, index=pairs.index)
    return out


def _prepare(net: RailNetwork, model: CostModel, pairs: pd.DataFrame, max_km: float):
    idx = {k: i for i, k in enumerate(model.node_ids)}
    p = pairs[(pairs.dist_km <= max_km) & (pairs.dist_km >= 30)].copy()
    p["ia"] = p.a.map(idx)
    p["ib"] = p.b.map(idx)
    p["y"] = p.direct.astype(int)
    seg = net.segment_edges
    g = csr_matrix((seg.time_min.to_numpy(), (seg.u.map(idx), seg.v.map(idx))), shape=(model.n, model.n))
    T = dijkstra(g, directed=False, indices=model.od)
    od_pos = {node: r for r, node in enumerate(model.od)}
    p["track_min"] = [T[od_pos[a], b] for a, b in zip(p.ia, p.ib)]
    p["track_min"] = p["track_min"].replace(np.inf, 3000.0)
    # Only daily-or-better links count as edges for the topology features.
    strong = net.service_edges[net.service_edges.trains_per_week >= 7]
    eu, ev = strong.u.map(idx).to_numpy(), strong.v.map(idx).to_numpy()
    return p.reset_index(drop=True), eu, ev


def evaluate(net: RailNetwork, model: CostModel, pairs: pd.DataFrame, max_km: float = 1200.0,
             folds: int = 5, seed: int = 0) -> dict:
    p, eu, ev = _prepare(net, model, pairs, max_km)
    y = p.y.to_numpy()
    models = {
        "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                                    random_state=seed),
    }
    keys = [f"{m} ({fs})" for fs in FEATURE_SETS for m in models]
    oof = {k: np.zeros(len(p)) for k in [*keys, "gravity_baseline", "adamic_adar_baseline"]}
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for train, test in skf.split(p, y):
        test_links = {(min(a, b), max(a, b)) for a, b, t in zip(p.ia[test], p.ib[test], y[test]) if t}
        keep = [(a, b) not in test_links and (b, a) not in test_links for a, b in zip(eu, ev)]
        nb = _neighbor_sets(eu[keep], ev[keep], model.n)
        X = pair_features(p, nb, p.track_min.to_numpy())
        for fs, cols in FEATURE_SETS.items():
            for name, make in models.items():
                clf = make().fit(X.iloc[train][cols], y[train])
                oof[f"{name} ({fs})"][test] = clf.predict_proba(X.iloc[test][cols])[:, 1]
        oof["gravity_baseline"][test] = (X.log_pop_prod - 2 * X.log_dist).iloc[test]
        oof["adamic_adar_baseline"][test] = X.adamic_adar.iloc[test]
    scores = pd.DataFrame([{"model": m, "roc_auc": roc_auc_score(y, s), "avg_precision": average_precision_score(y, s)}
                           for m, s in oof.items()]).sort_values("roc_auc", ascending=False)
    curves = {m: roc_curve(y, s) for m, s in oof.items()}
    return {"scores": scores.reset_index(drop=True), "curves": curves, "positives": int(y.sum()), "pairs": len(p)}


def missing_links(net: RailNetwork, model: CostModel, pairs: pd.DataFrame, max_km: float = 1200.0,
                  top: int = 20, seed: int = 0, features: str = "full") -> tuple[pd.DataFrame, pd.Series]:
    """Fit on all pairs; return the highest-probability pairs without a daily direct train,
    plus standardised logistic-regression coefficients for interpretation."""
    cols_x = FEATURE_SETS[features]
    p, eu, ev = _prepare(net, model, pairs, max_km)
    X = pair_features(p, _neighbor_sets(eu, ev, model.n), p.track_min.to_numpy())
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)
    clf.fit(X[cols_x], p.y)
    p["prob_direct"] = clf.predict_proba(X[cols_x])[:, 1]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[cols_x], p.y)
    coefs = pd.Series(lr[-1].coef_[0], index=cols_x).sort_values(key=np.abs, ascending=False)
    cols = ["name_a", "name_b", "country_a", "country_b", "dist_km", "pop_a", "pop_b", "trains_per_week",
            "journey_min", "trains_needed", "track_min", "prob_direct"]
    miss = p[~p.direct].sort_values("prob_direct", ascending=False).head(top)[cols].reset_index(drop=True)
    return miss, coefs
