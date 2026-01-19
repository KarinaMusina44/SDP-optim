# Question 3 — Explications de type (m-1)

## Partie du sujet (définition)
Dans l’affirmation $x \succ y$, un **trade-off de type (m-1)** ($m\in\mathbb{N}^*$) est une paire $(\{P_1,\dots,P_m\}, C)$ où :
- $P_1,\dots,P_m \in \mathrm{pros}(x,y)$,
- $C \in \mathrm{cons}(x,y)$,
- et la somme des contributions est **positive** :
$$
\sum_{i=1}^{m}\mathrm{contrib}(P_i) + \mathrm{contrib}(C) > 0.
$$

Une **explication de type (m-1)** est un ensemble
$$
E=\{(P^{(1)},C_1),\dots,(P^{(\ell)},C_\ell)\}
$$
de trade-offs disjoints tel que :
- les ensembles de pros $P^{(k)}$ sont disjoints (aucun "pro" réutilisé),
- et $\bigcup_{k=1}^{\ell}\{C_k\}=\mathrm{cons}(x,y)$ (chaque "con" est expliqué).

## Objectif
Formuler et implémenter un programme d’optimisation (avec Gurobi) qui :
1) calcule une explication (m-1) si elle existe ;
2) sinon, retourne un certificat pratique de non-existence (infeasibilité + IIS).

## Modélisation MILP (m-1)

### Variables
On affecte des "pros" à chaque "con" :

- $a_{p,c}\in\{0,1\}$ : vaut 1 si le pro $p\in\mathrm{pros}$ est utilisé pour compenser le con $c\in\mathrm{cons}$.

### Contraintes
1) **Un pro au plus une fois** (disjonction des ensembles de pros) :
$$
\sum_{c\in\mathrm{cons}} a_{p,c} \le 1 \quad \forall p\in\mathrm{pros}.
$$

2) **Chaque con doit être expliqué** : on impose une compensation strictement positive.
Comme les contributions sont entières (notes/poids entiers), $>0$ équivaut à $\ge 1$ :
$$
\mathrm{contrib}(c) + \sum_{p\in\mathrm{pros}} \mathrm{contrib}(p)\,a_{p,c} \ge 1 \quad \forall c\in\mathrm{cons}.
$$
(Optionnellement, on peut aussi imposer $\sum_{p} a_{p,c}\ge 1$, mais c’est déjà induit si $\mathrm{contrib}(c)<0$.)

### Objectif
On peut minimiser le **nombre total de pros utilisés** (explication la plus parcimonieuse) :
$$
\min \sum_{p\in\mathrm{pros}}\sum_{c\in\mathrm{cons}} a_{p,c}.
$$

Si le modèle est infeasible, on calcule un IIS via Gurobi.
