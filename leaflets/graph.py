"""Outils de théorie des graphes : composantes connexes et recherche automatique du seuil.

Recherche du seuil : on trie toutes les arêtes candidates par longueur et on les
ajoute une à une (algorithme de Kruskal / « single linkage » avec union-find).
On suit le nombre de « grosses » composantes (taille >= ``min_frac`` * N) et la
somme de leurs tailles. Les deux feuillets correspondent au plus long état à
exactement deux grosses composantes juste avant qu'elles ne fusionnent ; on
choisit le seuil au milieu de l'intervalle où leur couverture est maximale.
Un seul passage sur les arêtes donne donc la courbe complète « seuil -> partition ».
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@njit(cache=True)
def _find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


@njit(cache=True)
def _scan(n, i, j, dist, big):
    """Renvoie pour chaque arête fusionnante : (distance, n_big, couverture)."""
    parent = np.arange(n)
    size = np.ones(n, dtype=np.int64)
    m = len(i)
    ev_d = np.empty(m)
    ev_nbig = np.empty(m, dtype=np.int64)
    ev_cov = np.empty(m, dtype=np.int64)
    nbig = 0
    cov = 0
    k = 0
    for e in range(m):
        a = _find(parent, i[e])
        b = _find(parent, j[e])
        if a == b:
            continue
        sa, sb = size[a], size[b]
        if sa < sb:
            a, b = b, a
        parent[b] = a
        s = sa + sb
        size[a] = s
        if sa >= big:
            nbig -= 1
            cov -= sa
        if sb >= big:
            nbig -= 1
            cov -= sb
        if s >= big:
            nbig += 1
            cov += s
        ev_d[k] = dist[e]
        ev_nbig[k] = nbig
        ev_cov[k] = cov
        k += 1
    return ev_d[:k], ev_nbig[:k], ev_cov[:k]


def auto_cutoff(n, i, j, dist, cmax, min_frac=0.1, n_groups=2):
    """Seuil automatique pour obtenir ``n_groups`` grosses composantes.

    Renvoie (cutoff, info) ; ``cutoff`` vaut None si aucun état à ``n_groups``
    grosses composantes n'est trouvé.
    """
    order = np.argsort(dist, kind="stable")
    big = max(2, int(np.ceil(min_frac * n)))
    d, nbig, cov = _scan(n, i[order].astype(np.int64), j[order].astype(np.int64), dist[order], big)
    ok = nbig == n_groups
    if not ok.any():
        return None, {"reason": "pas d'état à %d composantes" % n_groups}
    # on considère le dernier « bloc » où l'on a n_groups grosses composantes
    # avant la fusion finale (nbig passe sous n_groups)
    after = np.flatnonzero(ok)
    last_ok = after[-1]
    merge = np.flatnonzero((np.arange(len(d)) > last_ok) & (nbig < n_groups))
    d_merge = d[merge[0]] if len(merge) else cmax
    # début du bloc contigu (en indices d'événements) se terminant à last_ok
    blk = last_ok
    while blk - 1 >= 0 and ok[blk - 1]:
        blk -= 1
    seg = np.arange(blk, last_ok + 1)
    kstar = seg[np.argmax(cov[seg])]
    # premier événement atteignant la couverture max
    kstar = seg[np.flatnonzero(cov[seg] == cov[kstar])[0]]
    cutoff = 0.5 * (d[kstar] + d_merge)
    return float(cutoff), {"d_start": float(d[kstar]), "d_merge": float(d_merge),
                           "coverage": float(cov[kstar] / n)}


def components(n, i, j, mask=None):
    if mask is not None:
        i, j = i[mask], j[mask]
    g = coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def largest_groups(labels, n_groups=2):
    """Réétiquette : 0..n_groups-1 pour les plus grosses composantes, -1 sinon."""
    counts = np.bincount(labels)
    top = np.argsort(counts)[::-1][:n_groups]
    out = np.full(len(labels), -1, dtype=np.int64)
    for k, c in enumerate(top):
        out[labels == c] = k
    return out
