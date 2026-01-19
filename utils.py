import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd


def explain_one_to_one(x_vec, y_vec, weights, criteria=None, eps_zero=0.0, gurobi_output=False):
    """
    Explication 1-1 pour x ≻ y.

    Inputs
    ------
    x_vec, y_vec, weights : array-like (même longueur)
    criteria : list[str] ou None (sinon ["k0","k1",...])
    eps_zero : tolérance pour considérer d_k = 0
    gurobi_output : afficher logs Gurobi

    Output
    ------
    dict avec:
      feasible (bool)
      d (dict crit->float)
      pros, cons, neutral (list[str])
      admissible_pairs (list[(p,c)])
      selected_pairs (list[(p,c)]) si feasible
      iis_constraints (list[str]) si infeasible
    """
    x = np.asarray(x_vec, dtype=float).reshape(-1)
    y = np.asarray(y_vec, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)

    if not (len(x) == len(y) == len(w)):
        raise ValueError(
            f"Length mismatch: len(x)={len(x)}, len(y)={len(y)}, len(w)={len(w)}")

    n = len(x)
    if criteria is None:
        crit = [f"k{i}" for i in range(n)]
    else:
        if len(criteria) != n:
            raise ValueError(
                f"criteria length mismatch: len(criteria)={len(criteria)} != {n}")
        crit = list(criteria)

    # d_k = w_k (x_k - y_k)
    d_vals = w * (x - y)
    d = {k: float(v) for k, v in zip(crit, d_vals)}

    # buckets
    def bucket(val):
        if abs(val) <= eps_zero:
            return 0
        return 1 if val > 0 else -1

    pros = [k for k in crit if bucket(d[k]) > 0]
    cons = [k for k in crit if bucket(d[k]) < 0]
    neutral = [k for k in crit if bucket(d[k]) == 0]

    # Cas simples
    if len(cons) == 0:
        return {
            "feasible": True,
            "d": d,
            "pros": pros,
            "cons": cons,
            "neutral": neutral,
            "admissible_pairs": [],
            "selected_pairs": [],
            "iis_constraints": None,
            "meta": {"note": "No cons -> empty explanation."},
        }

    if len(pros) == 0:
        return {
            "feasible": False,
            "d": d,
            "pros": pros,
            "cons": cons,
            "neutral": neutral,
            "admissible_pairs": [],
            "selected_pairs": [],
            "iis_constraints": ["No pros available to cover cons."],
            "meta": {"note": "pros is empty while cons is non-empty."},
        }

    # Paires admissibles
    A = [(p, c) for p in pros for c in cons if (d[p] + d[c]) > 0]
    if len(A) == 0:
        return {
            "feasible": False,
            "d": d,
            "pros": pros,
            "cons": cons,
            "neutral": neutral,
            "admissible_pairs": [],
            "selected_pairs": [],
            "iis_constraints": ["No admissible pairs (p,c) with d[p] + d[c] > 0."],
            "meta": {"note": "Admissible set A is empty."},
        }

    # Gurobi
    m = gp.Model("explanation_1_1")
    m.Params.OutputFlag = 1 if gurobi_output else 0

    xvar = m.addVars(A, vtype=GRB.BINARY, name="x")

    # cover cons exactly once
    for c in cons:
        m.addConstr(
            gp.quicksum(xvar[p, c] for p in pros if (p, c) in xvar) == 1,
            name=f"cover_cons[{c}]",
        )

    # each pro at most once
    for p in pros:
        m.addConstr(
            gp.quicksum(xvar[p, c] for c in cons if (p, c) in xvar) <= 1,
            name=f"use_pro_at_most_once[{p}]",
        )

    m.setObjective(0.0, GRB.MINIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        selected = [(p, c) for (p, c) in A if xvar[p, c].X > 0.5]
        return {
            "feasible": True,
            "d": d,
            "pros": pros,
            "cons": cons,
            "neutral": neutral,
            "admissible_pairs": A,
            "selected_pairs": selected,
            "iis_constraints": None,
            "meta": {"length": len(selected)},
        }

    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        iis = [constr.ConstrName for constr in m.getConstrs()
               if constr.IISConstr]
        return {
            "feasible": False,
            "d": d,
            "pros": pros,
            "cons": cons,
            "neutral": neutral,
            "admissible_pairs": A,
            "selected_pairs": [],
            "iis_constraints": iis,
            "meta": {"note": "Model infeasible; IIS returned."},
        }

    return {
        "feasible": False,
        "d": d,
        "pros": pros,
        "cons": cons,
        "neutral": neutral,
        "admissible_pairs": A,
        "selected_pairs": [],
        "iis_constraints": [f"Unexpected Gurobi status: {m.Status}"],
        "meta": {"gurobi_status": m.Status},
    }


def weighted_sum_ranking(df: pd.DataFrame, weights: pd.Series, *, normalize_weights: bool = False, score_col: str = "score"):
    """
    Score(i) = sum_{j in weights.index} weights[j] * df.loc[i, j]
    - df: index = alternatives (stations)
    - weights: pd.Series indexée par les colonnes à scorer (critères)
    """
    if not isinstance(weights, pd.Series):
        raise TypeError(
            "weights must be a pandas Series indexed by criterion columns.")

    # On score uniquement les colonnes présentes dans weights.index
    crit_cols = list(weights.index)

    missing_in_df = [c for c in crit_cols if c not in df.columns]
    if missing_in_df:
        raise ValueError(
            f"These weight columns are missing in df: {missing_in_df}")

    w = weights.loc[crit_cols].astype(float)

    if normalize_weights:
        s = float(w.sum())
        if s == 0:
            raise ValueError("Sum of weights is 0, can't normalize.")
        w = w / s

    X = df[crit_cols].apply(pd.to_numeric, errors="coerce")
    scores = X.mul(w, axis=1).sum(axis=1)

    ranked_df = df.copy()
    ranked_df[score_col] = scores
    ranked_df = ranked_df.sort_values(score_col, ascending=False)

    return ranked_df, w
