import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as transforms
from collections import defaultdict

import re


def plot_strategy_graph(
    stations_ranked,
    results,
    station_name_map=None,          # dict: id -> "Odéon (Ligne 4)"
    title="Graphe des comparaisons",
    show_gray_neighbors=True,
    alternate_sides=True,
    figsize=(11, 10),
    left_x_rank=-4.4,               # où écrire le rang (tout à gauche)
    left_x_name=-4.2,               # où écrire le nom (juste à droite du rang)
):
    """
    - Cercle: affiche l'id (station) au centre
    - À gauche: affiche le rang + le nom (si fourni)
    """
    pos = {node: (0.0, -i) for i, node in enumerate(stations_ranked)}
    n = len(stations_ranked)
    rank = {node: i for i, node in enumerate(stations_ranked)}

    # arêtes tentées
    attempted = {}  # (u,v) -> feasible bool
    for res in results:
        u, v = res.get("x"), res.get("y")
        if u is None or v is None:
            continue
        attempted[(u, v)] = bool(res.get("feasible", False))

    # arêtes grises: voisins du ranking non tentés
    gray_edges = []
    if show_gray_neighbors:
        for i in range(n - 1):
            u, v = stations_ranked[i], stations_ranked[i + 1]
            if (u, v) not in attempted:
                gray_edges.append((u, v))

    # niveaux par noeud et par côté
    out_levels_right = defaultdict(int)
    out_levels_left = defaultdict(int)
    edge_level = {}  # (u,v) -> (side, lvl)

    draw_order = gray_edges + list(attempted.keys())
    for (u, v) in draw_order:
        if alternate_sides:
            side = "right" if (rank[u] % 2 == 0) else "left"
        else:
            side = "right"

        if side == "right":
            lvl = out_levels_right[u]
            out_levels_right[u] += 1
        else:
            lvl = out_levels_left[u]
            out_levels_left[u] += 1

        edge_level[(u, v)] = (side, lvl)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.axis("off")

    # noeuds (cercles)
    xs = [pos[node][0] for node in stations_ranked]
    ys = [pos[node][1] for node in stations_ranked]
    ax.scatter(xs, ys, s=520, zorder=5)  # cercles

    # texte AU CENTRE des cercles = id station
    for node in stations_ranked:
        x, y = pos[node]
        ax.text(x, y, str(node), ha="center",
                va="center", fontsize=11, zorder=6)

    # rang + nom tout à gauche, aligné sur le cercle
    for i, node in enumerate(stations_ranked, start=1):
        y = pos[node][1]

        # rang
        ax.text(left_x_rank, y, f"{i}", ha="right",
                va="center", fontsize=10, zorder=6)

        # nom
        if station_name_map is not None and node in station_name_map:
            name = str(station_name_map[node])
        else:
            name = ""  # fallback si pas fourni

        if name:
            ax.text(left_x_name, y, name, ha="left",
                    va="center", fontsize=10, zorder=6)

    def draw_curved_arrow(u, v, color, alpha=1.0, lw=2.0, linestyle="-", zorder=2):
        (x1, y1) = pos[u]
        (x2, y2) = pos[v]
        side, lvl = edge_level[(u, v)]

        base = 0.9
        step = 0.22
        side_sign = 1 if side == "right" else -1
        x_offset = (base + step * lvl) * side_sign

        start = (x1 + 0.06 * side_sign, y1)
        end = (x2 + 0.06 * side_sign, y2)

        rad_base = 0.25 + 0.03 * lvl
        rad = rad_base if side == "right" else -rad_base

        arrow = FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder
        )

        arrow.set_transform(
            ax.transData + transforms.Affine2D().translate(x_offset, 0))
        ax.add_patch(arrow)

    # gris d'abord
    for (u, v) in gray_edges:
        draw_curved_arrow(u, v, color="gray", alpha=0.30,
                          lw=1.4, linestyle="--", zorder=1)

    # tentées ensuite
    for (u, v), ok in attempted.items():
        if ok:
            draw_curved_arrow(u, v, color="green", alpha=0.9,
                              lw=2.5, linestyle="-", zorder=3)
        else:
            draw_curved_arrow(u, v, color="red", alpha=0.9,
                              lw=2.5, linestyle="-", zorder=3)

    # limites (ajuste si besoin)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-n - 1, 1)
    plt.show()


def _edges_from_mixed_results(results, method=None, *, prefer_feasible=True):
    """
    Extrait un mapping (u,v)->feasible depuis une liste results mixte.
    - method=None : prend tout
    - method="1-1" ou "1-m" : filtre sur res["method"]
    - prefer_feasible=True : si une même paire apparaît plusieurs fois, on garde vert
      dès qu'il y a au moins un succès dans ce sous-ensemble.
      (Très utile avec fallback.)
    """
    edge_map = {}
    for res in results:
        if method is not None and res.get("method") != method:
            continue

        u, v = res.get("x"), res.get("y")
        if u is None or v is None:
            continue

        ok = bool(res.get("feasible", False))
        if prefer_feasible:
            edge_map[(u, v)] = edge_map.get((u, v), False) or ok
        else:
            edge_map[(u, v)] = ok  # last wins
    return edge_map


def plot_strategy_graph_ax(
    ax,
    stations_ranked,
    edge_status,                 # dict (u,v)->bool
    station_name_map=None,
    title="Graphe des comparaisons",
    show_gray_neighbors=True,
    alternate_sides=True,
    left_x_rank=-4.4,
    left_x_name=-4.2,
):
    """
    Plot sur un ax donné.
    edge_status: dict unique par paire (u,v) -> feasible bool
    """
    pos = {node: (0.0, -i) for i, node in enumerate(stations_ranked)}
    n = len(stations_ranked)
    rank = {node: i for i, node in enumerate(stations_ranked)}

    # voisins non tentés (gris)
    gray_edges = []
    if show_gray_neighbors:
        for i in range(n - 1):
            u, v = stations_ranked[i], stations_ranked[i + 1]
            if (u, v) not in edge_status:
                gray_edges.append((u, v))

    # routing anti-overlap (par noeud source u)
    out_levels_right = defaultdict(int)
    out_levels_left = defaultdict(int)
    edge_level = {}  # (u,v)->(side,lvl)

    draw_order = gray_edges + list(edge_status.keys())
    for (u, v) in draw_order:
        if alternate_sides:
            side = "right" if (rank[u] % 2 == 0) else "left"
        else:
            side = "right"

        if side == "right":
            lvl = out_levels_right[u]
            out_levels_right[u] += 1
        else:
            lvl = out_levels_left[u]
            out_levels_left[u] += 1

        edge_level[(u, v)] = (side, lvl)

    ax.set_title(title)
    ax.axis("off")

    # noeuds
    xs = [pos[node][0] for node in stations_ranked]
    ys = [pos[node][1] for node in stations_ranked]
    ax.scatter(xs, ys, s=520, zorder=5, edgecolors="black", linewidths=0.8)

    for node in stations_ranked:
        x, y = pos[node]
        ax.text(x, y, str(node), ha="center",
                va="center", fontsize=11, zorder=6)

    # rang + nom
    for i, node in enumerate(stations_ranked, start=1):
        y = pos[node][1]
        ax.text(left_x_rank, y, f"{i}", ha="right",
                va="center", fontsize=10, zorder=6)

        name = ""
        if station_name_map is not None and node in station_name_map:
            name = str(station_name_map[node])
        if name:
            ax.text(left_x_name, y, name, ha="left",
                    va="center", fontsize=10, zorder=6)

    def draw_curved_arrow(u, v, *, color, alpha=1.0, lw=2.0, linestyle="-", zorder=2):
        (x1, y1) = pos[u]
        (x2, y2) = pos[v]
        side, lvl = edge_level[(u, v)]

        base = 0.9
        step = 0.22
        side_sign = 1 if side == "right" else -1
        x_offset = (base + step * lvl) * side_sign

        start = (x1 + 0.06 * side_sign, y1)
        end = (x2 + 0.06 * side_sign, y2)

        rad_base = 0.25 + 0.03 * lvl
        rad = rad_base if side == "right" else -rad_base

        arrow = FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder
        )
        arrow.set_transform(
            ax.transData + transforms.Affine2D().translate(x_offset, 0))
        ax.add_patch(arrow)

    # gris
    for (u, v) in gray_edges:
        draw_curved_arrow(u, v, color="gray", alpha=0.30,
                          lw=1.4, linestyle="--", zorder=1)

    # tentées
    for (u, v), ok in edge_status.items():
        draw_curved_arrow(u, v, color=("green" if ok else "red"),
                          alpha=0.9, lw=2.5, zorder=3)

    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-n - 1, 1)


def plot_three_plots_from_descend_results(
    stations_ranked,
    results_mixed,                 # sortie de run_strategy_descend_with_fallback_1m
    station_name_map=None,
    figsize=(22, 10),
    show_gray_neighbors=True,
    alternate_sides=True,
    prefer_feasible=True,          # si une paire a eu un succès au moins une fois => vert
):
    """
    3 plots côte à côte:
      1) 1-1 only
      2) 1-m only
      3) résultat fusionné : priorité 1-m, sinon 1-1
    => aucune superposition 1-m/1-1 dans le plot résultat car 1 seule flèche par paire.
    """
    edges_11 = _edges_from_mixed_results(
        results_mixed, method="1-1", prefer_feasible=prefer_feasible)
    edges_1m = _edges_from_mixed_results(
        results_mixed, method="1-m", prefer_feasible=prefer_feasible)

    # fusion avec priorité 1-m
    merged = dict(edges_11)
    merged.update(edges_1m)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_strategy_graph_ax(
        axes[0], stations_ranked, edges_11,
        station_name_map=station_name_map,
        title="1-1",
        show_gray_neighbors=show_gray_neighbors,
        alternate_sides=alternate_sides,
    )

    plot_strategy_graph_ax(
        axes[1], stations_ranked, edges_1m,
        station_name_map=station_name_map,
        title="1-m",
        show_gray_neighbors=show_gray_neighbors,
        alternate_sides=alternate_sides,
    )

    plot_strategy_graph_ax(
        axes[2], stations_ranked, merged,
        station_name_map=station_name_map,
        title="Résultat",
        show_gray_neighbors=show_gray_neighbors,
        alternate_sides=alternate_sides,
    )

    plt.tight_layout()
    plt.show()


def edges_from_results(results, method=None):
    """
    Retourne un dict edge_status: (u,v)->bool (feasible)
    Si une même arête apparaît plusieurs fois, on met True si au moins une tentative est True.
    """
    edge_status = {}
    for r in results:
        if method is not None and r.get("method") != method:
            continue
        u, v = r.get("x"), r.get("y")
        if u is None or v is None:
            continue
        ok = bool(r.get("feasible", False))
        edge_status[(u, v)] = edge_status.get((u, v), False) or ok
    return edge_status


def plot_strategy_graph_on_ax(
    ax,
    stations_ranked,
    edge_status,                 # dict (u,v)->bool
    station_name_map=None,
    title="",
    show_gray_neighbors=True,
    alternate_sides=True,
    left_x_rank=-4.4,
    left_x_name=-4.2,
    edge_labels=None,            # dict (u,v)->str
    show_edge_labels=False,
):
    pos = {node: (0.0, -i) for i, node in enumerate(stations_ranked)}
    n = len(stations_ranked)
    rank = {node: i for i, node in enumerate(stations_ranked)}

    if edge_labels is None:
        edge_labels = {}

    # Arêtes grises = voisins du ranking non présents dans edge_status
    gray_edges = []
    if show_gray_neighbors:
        for i in range(n - 1):
            u, v = stations_ranked[i], stations_ranked[i + 1]
            if (u, v) not in edge_status:
                gray_edges.append((u, v))

    # Routage anti-chevauchement (par noeud source)
    out_levels_right = defaultdict(int)
    out_levels_left = defaultdict(int)
    edge_level = {}

    draw_order = gray_edges + list(edge_status.keys())
    for (u, v) in draw_order:
        if alternate_sides:
            side = "right" if (rank[u] % 2 == 0) else "left"
        else:
            side = "right"

        if side == "right":
            lvl = out_levels_right[u]
            out_levels_right[u] += 1
        else:
            lvl = out_levels_left[u]
            out_levels_left[u] += 1

        edge_level[(u, v)] = (side, lvl)

    ax.set_title(title)
    ax.axis("off")

    # Noeuds
    xs = [pos[node][0] for node in stations_ranked]
    ys = [pos[node][1] for node in stations_ranked]
    ax.scatter(xs, ys, s=520, zorder=5, edgecolors="black", linewidths=0.8)

    # id au centre
    for node in stations_ranked:
        x, y = pos[node]
        ax.text(x, y, str(node), ha="center",
                va="center", fontsize=11, zorder=6)

    # rang + nom
    for i, node in enumerate(stations_ranked, start=1):
        y = pos[node][1]
        ax.text(left_x_rank, y, f"{i}", ha="right",
                va="center", fontsize=10, zorder=6)
        if station_name_map is not None and node in station_name_map:
            ax.text(left_x_name, y, str(
                station_name_map[node]), ha="left", va="center", fontsize=10, zorder=6)

    def draw_arrow(u, v, color, alpha=1.0, lw=2.0, linestyle="-", zorder=2, label=None):
        (x1, y1) = pos[u]
        (x2, y2) = pos[v]
        side, lvl = edge_level[(u, v)]

        base = 0.9
        step = 0.22
        side_sign = 1 if side == "right" else -1
        x_offset = (base + step * lvl) * side_sign

        start = (x1 + 0.06 * side_sign, y1)
        end = (x2 + 0.06 * side_sign, y2)

        rad_base = 0.25 + 0.03 * lvl
        rad = rad_base if side == "right" else -rad_base

        arrow = FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder
        )

        tr = ax.transData + transforms.Affine2D().translate(x_offset, 0)
        arrow.set_transform(tr)
        ax.add_patch(arrow)

        if show_edge_labels and label:
            mx = (start[0] + end[0]) / 2.0
            my = (start[1] + end[1]) / 2.0
            ax.text(mx, my, str(label), transform=tr, ha="center", va="center",
                    fontsize=9, zorder=zorder+1)

    # Gris
    for (u, v) in gray_edges:
        draw_arrow(u, v, color="gray", alpha=0.30,
                   lw=1.4, linestyle="--", zorder=1)

    # Tentées
    for (u, v), ok in edge_status.items():
        lab = edge_labels.get((u, v), None)
        draw_arrow(u, v, color=("green" if ok else "red"),
                   alpha=0.9, lw=2.5, zorder=3, label=lab)

    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-n - 1, 1)


def plot_five_panels(
    stations_ranked,
    results_mixed,
    station_name_map=None,
    show_gray_neighbors=True,
    alternate_sides=True,
    figsize=(34, 10),
    priority=("mix", "m-1", "1-m", "1-1"),  # priorité dans le MERGE final
    # nom du champ method pour le mixte dans results
    mix_method_name="mix",
):
    """
    5 panneaux:
      [0] 1-1
      [1] 1-m
      [2] m-1
      [3] mix (1-m ∪ m-1)
      [4] fusion des arêtes selon priorité (edge_labels = méthode gagnante)

    priority: du plus prioritaire -> moins prioritaire
              ex ("mix","m-1","1-m","1-1")
    mix_method_name: si dans tes results tu stockes le mix sous "mix" (recommandé),
                     ou autre string.
    """

    # --- arêtes par méthode
    edges_11 = edges_from_results(results_mixed, method="1-1")
    edges_1m = edges_from_results(results_mixed, method="1-m")
    edges_m1 = edges_from_results(results_mixed, method="m-1")
    edges_mix = edges_from_results(results_mixed, method=mix_method_name)

    edges_by = {
        "1-1": edges_11,
        "1-m": edges_1m,
        "m-1": edges_m1,
        mix_method_name: edges_mix,
    }

    # --- merge avec priorité
    # on applique du moins prioritaire au plus prioritaire
    apply_order = list(priority)[::-1]
    merged = {}
    merged_labels = {}

    for meth in apply_order:
        if meth not in edges_by:
            raise ValueError(
                f"priority contains '{meth}' not found in edges_by={list(edges_by.keys())} "
                f"(check mix_method_name)"
            )
        for e, ok in edges_by[meth].items():
            merged[e] = ok
            merged_labels[e] = meth

    # --- plot
    fig, axes = plt.subplots(1, 5, figsize=figsize)

    plot_strategy_graph_on_ax(
        axes[0], stations_ranked, edges_11,
        station_name_map=station_name_map,
        title="1-1",
        show_gray_neighbors=show_gray_neighbors,
        alternate_sides=alternate_sides,
        show_edge_labels=False,
    )

    plot_strategy_graph_on_ax(
        axes[1], stations_ranked, edges_1m,
        station_name_map=station_name_map,
        title="1-m",
        show_gray_neighbors=show_gray_neighbors,
        alternate_sides=alternate_sides,
        show_edge_labels=False,
    )

    plot_strategy_graph_on_ax(
        axes[2], stations_ranked, edges_m1,
        station_name_map=station_name_map,
        title="m-1",
        show_gray_neighbors=show_gray_neighbors,
        alternate_sides=alternate_sides,
        show_edge_labels=False,
    )

    plot_strategy_graph_on_ax(
        axes[3], stations_ranked, edges_mix,
        station_name_map=station_name_map,
        title=mix_method_name,
        show_gray_neighbors=show_gray_neighbors,
        alternate_sides=alternate_sides,
        show_edge_labels=False,
    )

    plot_strategy_graph_on_ax(
        axes[4], stations_ranked, merged,
        station_name_map=station_name_map,
        title=f"Résultat",
        show_gray_neighbors=show_gray_neighbors,
        alternate_sides=alternate_sides,
        edge_labels=merged_labels,
        # show_edge_labels=True,  # labels = "1-1" / "1-m" / "m-1" / "mix"
    )

    plt.tight_layout()
    plt.show()


def _base_note(note: str) -> str:
    if note is None:
        return ""
    return re.sub(r"^(fallback\s+)+", "", str(note)).strip()


def build_parent_tree_from_results(results, stations_ranked=None):
    """
    Construit un arbre parent->enfant.
    Règle: pour chaque enfant y, on prend le 1er (x,y) feasible rencontré dans results.
    Optionnel: si stations_ranked est fourni, on ignore les arêtes qui violent l'ordre du ranking
              (ex: parent doit être "au-dessus": rank[parent] < rank[child]).
    Retour:
      parent_of : dict child -> parent
      method_of : dict (parent,child) -> method
    """
    parent_of = {}
    method_of = {}

    rank = None
    if stations_ranked is not None:
        rank = {n: i for i, n in enumerate(stations_ranked)}

    for r in results:
        if not r.get("feasible", False):
            continue
        x = r.get("x", None)
        y = r.get("y", None)
        if x is None or y is None:
            continue

        # option: respecter la direction "du haut vers le bas" dans le ranking
        if rank is not None and x in rank and y in rank:
            if rank[x] >= rank[y]:
                continue

        # arbre => un seul parent par enfant
        if y in parent_of:
            continue

        parent_of[y] = x
        method_of[(x, y)] = r.get("method", "")
    return parent_of, method_of


def compute_depths_from_parent(parent_of, stations_ranked):
    """
    depth[root]=0, depth[child]=depth[parent]+1
    Si un nœud n'a pas de parent: root.
    """
    depth = {n: 0 for n in stations_ranked}

    def get_depth(n, visiting=None):
        if visiting is None:
            visiting = set()
        if n in visiting:
            # cycle improbable; on casse
            return 0
        visiting.add(n)

        if n not in parent_of:
            d = 0
        else:
            p = parent_of[n]
            d = get_depth(p, visiting) + 1
        visiting.remove(n)
        return d

    for n in stations_ranked:
        depth[n] = get_depth(n)

    return depth


def plot_tree_graph_on_ax(
    ax,
    stations_ranked,
    parent_of,                   # dict child->parent
    method_of=None,              # dict (parent,child)->str (optionnel)
    station_name_map=None,
    title="Arbre (résultat)",
    alternate_sides=True,
    x_step=2.6,                  # distance entre profondeurs
    left_x_rank=None,
    left_x_name=None,
    show_edge_labels=True,
):
    """
    Plot en arbre:
      - Nodes à y = -rank, x = depth * x_step
      - Edges = parent -> child (un parent par node)
      - Labels d'arêtes = méthode (1-1/1-m/m-1/mix)
    """
    if method_of is None:
        method_of = {}

    n = len(stations_ranked)
    rank = {node: i for i, node in enumerate(stations_ranked)}

    depth = compute_depths_from_parent(parent_of, stations_ranked)
    max_depth = max(depth.values()) if depth else 0

    pos = {node: (depth[node] * x_step, -i)
           for i, node in enumerate(stations_ranked)}

    # Position texte à gauche (si pas donné)
    min_x = 0.0
    if left_x_rank is None:
        left_x_rank = min_x - 1.6
    if left_x_name is None:
        left_x_name = min_x - 1.4

    ax.set_title(title)
    ax.axis("off")

    # Noeuds
    xs = [pos[node][0] for node in stations_ranked]
    ys = [pos[node][1] for node in stations_ranked]
    ax.scatter(xs, ys, s=520, zorder=5, edgecolors="black", linewidths=0.8)

    # id au centre
    for node in stations_ranked:
        x, y = pos[node]
        ax.text(x, y, str(node), ha="center",
                va="center", fontsize=11, zorder=6)

    # rang + nom
    for i, node in enumerate(stations_ranked, start=1):
        y = pos[node][1]
        ax.text(left_x_rank, y, f"{i}", ha="right",
                va="center", fontsize=10, zorder=6)
        if station_name_map is not None and node in station_name_map:
            ax.text(left_x_name, y, str(
                station_name_map[node]), ha="left", va="center", fontsize=10, zorder=6)

    # Routage anti-chevauchement (par noeud source)
    out_levels_right = defaultdict(int)
    out_levels_left = defaultdict(int)
    edge_level = {}

    edges = []
    for child, parent in parent_of.items():
        if parent is None:
            continue
        # direction parent -> child
        edges.append((parent, child))

    # ordonner un peu: parent en haut d'abord (cosmétique)
    edges.sort(key=lambda e: (rank.get(e[0], 10**9), rank.get(e[1], 10**9)))

    for (u, v) in edges:
        if alternate_sides:
            side = "right" if (rank[u] % 2 == 0) else "left"
        else:
            side = "right"

        if side == "right":
            lvl = out_levels_right[u]
            out_levels_right[u] += 1
        else:
            lvl = out_levels_left[u]
            out_levels_left[u] += 1
        edge_level[(u, v)] = (side, lvl)

    def draw_arrow(u, v, color="green", alpha=0.95, lw=2.6, linestyle="-", zorder=3, label=None):
        (x1, y1) = pos[u]
        (x2, y2) = pos[v]
        side, lvl = edge_level[(u, v)]

        # un peu d'offset latéral pour éviter superpositions d'arêtes sortantes
        base = 0.00
        step = 0.18
        side_sign = 1 if side == "right" else -1
        x_offset = (base + step * lvl) * side_sign

        start = (x1 + 0.10, y1)
        end = (x2 + 0.10, y2)

        # courbure légère
        rad_base = 0.18 + 0.03 * lvl
        rad = rad_base if side == "right" else -rad_base

        arrow = FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder
        )

        tr = ax.transData + transforms.Affine2D().translate(x_offset, 0)
        arrow.set_transform(tr)
        ax.add_patch(arrow)

        if show_edge_labels and label:
            t = 0.8  # <--- 0.25..0.45 selon ton goût
            mx = start[0] + 0.7 * (end[0] - start[0])
            my = start[1] + 0.9 * (end[1] - start[1])
            ax.text(mx, my, str(label), transform=tr, ha="center", va="center",
                    fontsize=15, zorder=zorder+1)

    # Edges (toutes vertes, car arbre = retenues)
    for (u, v) in edges:
        lab = method_of.get((u, v), None)
        draw_arrow(u, v, color="green", label=lab)

    # limites
    ax.set_xlim(min_x - 2.2, (max_depth * x_step) + 2.2)
    ax.set_ylim(-n - 1, 1)


def plot_results_as_tree_like_last_panel(
    stations_ranked,
    results,
    station_name_map=None,
    figsize=(16, 10),
    title="Résultat en arbre",
    x_step=2.6,
):
    parent_of, method_of = build_parent_tree_from_results(
        results, stations_ranked=stations_ranked)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_tree_graph_on_ax(
        ax,
        stations_ranked=stations_ranked,
        parent_of=parent_of,
        method_of=method_of,
        station_name_map=station_name_map,
        title=title,
        x_step=x_step,
        show_edge_labels=True,   # labels = method
    )
    plt.tight_layout()
    plt.show()
