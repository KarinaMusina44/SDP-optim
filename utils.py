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


def explain_one_to_many(
    x_vec,
    y_vec,
    weights,
    criteria=None,
    eps_zero=0.0,
    eps_pos=1e-9,
    gurobi_output=False,
    minimize_groups=True,
):
    """
    Explication (1-m) pour x ≻ y.

    Un "pour" p peut couvrir plusieurs "contre" c.
    On partitionne cons en groupes C_p associés à des pros p, avec condition:
        d[p] + sum_{c in C_p} d[c] > 0

    Inputs
    ------
    x_vec, y_vec, weights : array-like (même longueur)
    criteria : list[str] ou None
    eps_zero : tolérance pour considérer d_k = 0
    eps_pos : seuil strict de positivité (>= eps_pos quand groupe utilisé)
    gurobi_output : afficher logs Gurobi
    minimize_groups : minimiser le nb de pros utilisés (explication la plus courte)

    Output
    ------
    dict avec:
      feasible (bool)
      d (dict crit->float)
      pros, cons, neutral
      selected_groups (dict p -> list[c]) si feasible
      selected_pairs (list[(p,c)]) si feasible
      iis_constraints si infeasible
      meta (info)
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

    # contributions
    d_vals = w * (x - y)
    d = {k: float(v) for k, v in zip(crit, d_vals)}

    def bucket(val):
        if abs(val) <= eps_zero:
            return 0
        return 1 if val > 0 else -1

    pros = [k for k in crit if bucket(d[k]) > 0]
    cons = [k for k in crit if bucket(d[k]) < 0]
    neutral = [k for k in crit if bucket(d[k]) == 0]

    # cas simples
    if len(cons) == 0:
        return {
            "feasible": True,
            "d": d,
            "pros": pros,
            "cons": cons,
            "neutral": neutral,
            "selected_groups": {},
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
            "selected_groups": {},
            "selected_pairs": [],
            "iis_constraints": ["No pros available to cover cons."],
            "meta": {"note": "pros is empty while cons is non-empty."},
        }

    # modèle Gurobi
    m = gp.Model("explanation_1_m")
    m.Params.OutputFlag = 1 if gurobi_output else 0

    # variables:
    # x[p,c] = 1 si c assigné au groupe de p
    A = [(p, c) for p in pros for c in cons]
    xvar = m.addVars(A, vtype=GRB.BINARY, name="x")

    # y[p] = 1 si le pro p est utilisé (a au moins un con assigné)
    yvar = m.addVars(pros, vtype=GRB.BINARY, name="y")

    # C1 : chaque con couvert exactement une fois
    for c in cons:
        m.addConstr(
            gp.quicksum(xvar[p, c] for p in pros) == 1,
            name=f"cover_cons[{c}]",
        )

    # Lien: si x[p,c]=1 alors y[p]=1
    for (p, c) in A:
        m.addConstr(xvar[p, c] <= yvar[p], name=f"link[{p},{c}]")

    # C2 : validité des trade-offs quand p est utilisé
    # d[p] + sum_c d[c] x[p,c] >= eps_pos   si y[p]=1
    # via Big-M:  d[p] + sum_c d[c] x[p,c] >= eps_pos - M*(1 - y[p])
    # Choix d'un M sûr:
    # LHS min possible (en valeur) : d[p] + sum d[c] (si tout cons assigné) >= d[p] + sum d[c]
    # On veut relâcher jusqu'à très bas quand y=0.
    sum_abs_cons = sum(abs(d[c]) for c in cons)
    for p in pros:
        # borne très conservatrice
        M = abs(d[p]) + sum_abs_cons + abs(eps_pos) + 1.0
        m.addConstr(
            d[p] + gp.quicksum(d[c] * xvar[p, c]
                               for c in cons) >= eps_pos - M * (1 - yvar[p]),
            name=f"tradeoff_pos[{p}]",
        )

    # objectif: minimiser le nombre de groupes utilisés (optionnel mais recommandé)
    if minimize_groups:
        m.setObjective(gp.quicksum(yvar[p] for p in pros), GRB.MINIMIZE)
    else:
        m.setObjective(0.0, GRB.MINIMIZE)

    m.optimize()

    if m.Status == GRB.OPTIMAL:
        selected_pairs = [(p, c) for (p, c) in A if xvar[p, c].X > 0.5]
        groups = {}
        for p, c in selected_pairs:
            groups.setdefault(p, []).append(c)

        return {
            "feasible": True,
            "d": d,
            "pros": pros,
            "cons": cons,
            "neutral": neutral,
            "selected_groups": groups,
            "selected_pairs": selected_pairs,
            "iis_constraints": None,
            "meta": {"num_groups": len(groups), "num_pairs": len(selected_pairs)},
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
            "selected_groups": {},
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
        "selected_groups": {},
        "selected_pairs": [],
        "iis_constraints": [f"Unexpected Gurobi status: {m.Status}"],
        "meta": {"gurobi_status": m.Status},
    }


def pretty_print_explanation_1m(res, max_groups=None):
    """
    Affichage compact pour explain_one_to_many.
    """
    x, y = res.get("x"), res.get("y")
    head = f"{x} ≻ {y}" if (x is not None and y is not None) else "x ≻ y"
    print(f"\n=== Explication 1-m pour: {head} ===")
    print("feasible:", res["feasible"])
    print("pros:", res["pros"])
    print("cons:", res["cons"])
    print("neutral:", res["neutral"])

    if not res["feasible"]:
        print("\nInfaisable.")
        print("IIS / diagnostic:", res.get("iis_constraints"))
        return

    groups = res["selected_groups"]
    items = list(groups.items())
    if max_groups is not None:
        items = items[:max_groups]

    print("\nGroupes retenus :")
    for p, cs in items:
        dp = res["d"][p]
        sc = sum(res["d"][c] for c in cs)
        total = dp + sc
        cs_list = ", ".join(cs)
        print(
            f" - {p}({dp:+.3f}) couvre {{{cs_list}}}({sc:+.3f}) => somme={total:+.3f}")

    print("Longueur l (nb groupes) =", len(groups))
