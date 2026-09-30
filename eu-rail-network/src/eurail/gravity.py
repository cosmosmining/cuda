"""How connected are city pairs, and how does that relate to distance and size?

Builds a table of all OD-zone pairs and fits gravity-type models:

1. PPML (Poisson pseudo-maximum likelihood) of direct weekly trains on
   log populations, log distance and a same-country dummy. PPML is the standard
   gravity estimator when many pairs have zero flow.
2. Logit of "has a direct train at least daily".
3. OLS of log generalized journey time on log distance (elasticity < 1 means
   long trips are relatively faster, i.e. high-speed rail) plus the controls.

``exp(beta_same_country)`` measures the border effect: how many times more
direct service a domestic pair gets than an otherwise identical cross-border pair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .metrics import CostModel


def od_pair_table(nodes: pd.DataFrame, service: pd.DataFrame, model: CostModel, R_od: np.ndarray) -> pd.DataFrame:
    od_nodes = nodes.iloc[model.od].reset_index(drop=True)
    m = len(od_nodes)
    iu, ju = np.triu_indices(m, 1)
    J = model.journey_minutes(R_od)

    g = csr_matrix((np.ones(len(model.u)), (model.u, model.v)), shape=(model.n, model.n))
    hops = dijkstra(g, directed=False, unweighted=True, indices=model.od)[:, model.od]

    ids = od_nodes["node_id"].to_numpy()
    df = pd.DataFrame({
        "a": ids[iu], "b": ids[ju],
        "name_a": od_nodes["name"].to_numpy()[iu], "name_b": od_nodes["name"].to_numpy()[ju],
        "country_a": od_nodes["country"].to_numpy()[iu], "country_b": od_nodes["country"].to_numpy()[ju],
        "pop_a": od_nodes["population"].to_numpy()[iu], "pop_b": od_nodes["population"].to_numpy()[ju],
        "dist_km": model.dist_km[iu, ju],
        "journey_min": J[iu, ju],
        "trains_needed": hops[iu, ju],
        "demand": model.W[iu, ju] + model.W[ju, iu],
    })
    e = service.copy()
    key = lambda x, y: np.where(x < y, x + "|" + y, y + "|" + x)  # noqa: E731
    e["k"] = key(e.u.to_numpy(), e.v.to_numpy())
    df["k"] = key(df.a.to_numpy(), df.b.to_numpy())
    df = df.merge(e[["k", "trains_per_week", "time_median"]], on="k", how="left").drop(columns="k")
    df["trains_per_week"] = df["trains_per_week"].fillna(0.0)
    df["direct"] = df["trains_per_week"] >= 7
    df["same_country"] = (df["country_a"] == df["country_b"]).astype(int)
    df["log_pop_prod"] = np.log(df.pop_a.astype(float)) + np.log(df.pop_b.astype(float))
    df["log_dist"] = np.log(df.dist_km)
    df["eff_speed_kmh"] = df.dist_km / (df.journey_min / 60.0)
    return df


def _coef_table(res, name: str) -> pd.DataFrame:
    ci = res.conf_int()
    return pd.DataFrame({
        "model": name, "term": res.params.index, "coef": res.params.values, "std_err": res.bse.values,
        "p_value": res.pvalues.values, "ci_low": ci[0].values, "ci_high": ci[1].values,
    })


def fit_gravity_models(pairs: pd.DataFrame, min_km: float = 50.0, max_km: float = 1500.0) -> dict:
    d = pairs[(pairs.dist_km >= min_km) & (pairs.dist_km <= max_km)].copy()
    d["direct_i"] = d["direct"].astype(int)

    ppml = smf.glm("trains_per_week ~ log_pop_prod + log_dist + same_country", data=d,
                   family=sm.families.Poisson()).fit(cov_type="HC1")
    logit = smf.logit("direct_i ~ log_pop_prod + log_dist + same_country", data=d).fit(disp=0, cov_type="HC1")
    reach = d[np.isfinite(d.journey_min)].copy()
    reach["log_journey"] = np.log(reach.journey_min)
    ols = smf.ols("log_journey ~ log_dist + log_pop_prod + same_country", data=reach).fit(cov_type="HC1")

    reach["time_residual"] = ols.resid
    reach["expected_journey_min"] = np.exp(ols.fittedvalues)
    coefs = pd.concat([_coef_table(ppml, "ppml_direct_trains"), _coef_table(logit, "logit_direct_link"),
                       _coef_table(ols, "ols_log_journey_time")], ignore_index=True)
    stats = {
        "pairs_used": int(len(d)),
        "pairs_with_direct_link": int(d.direct.sum()),
        "ppml_pseudo_r2": float(1 - ppml.deviance / ppml.null_deviance),
        "logit_pseudo_r2": float(logit.prsquared),
        "ols_r2": float(ols.rsquared),
        "border_effect_trains": float(np.exp(ppml.params["same_country"])),
        "border_effect_link_odds": float(np.exp(logit.params["same_country"])),
        "distance_elasticity_trains": float(ppml.params["log_dist"]),
        "size_elasticity_trains": float(ppml.params["log_pop_prod"]),
        "distance_elasticity_time": float(ols.params["log_dist"]),
    }
    return {"coefficients": coefs, "stats": stats, "residuals": reach}


def underserved_pairs(residuals: pd.DataFrame, top: int = 20, min_km: float = 150.0) -> pd.DataFrame:
    """Pairs much slower than the model expects, ranked by demand-weighted excess time."""
    r = residuals[residuals.dist_km >= min_km].copy()
    r["excess_min"] = r.journey_min - r.expected_journey_min
    r["score"] = r.demand * r.excess_min.clip(lower=0)
    cols = ["name_a", "name_b", "dist_km", "journey_min", "expected_journey_min", "excess_min", "trains_needed", "trains_per_week", "score"]
    return r.sort_values("score", ascending=False).head(top)[cols].reset_index(drop=True)
