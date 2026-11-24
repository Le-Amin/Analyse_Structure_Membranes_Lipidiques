"""Version 3 : orientation des lipides + normale locale à chaque lipide.

1. Tête h_i (atome choisi automatiquement, V2) et orientation o_i = h_i - g_i,
   où g_i est le centre géométrique du lipide (rendu entier vis-à-vis des PBC).
2. Normale locale n_i : analyse en composantes principales (ACP) des têtes
   voisines (rayon ``r_normal``) de même orientation (o_i . o_j > 0) ; la normale
   est le vecteur propre de plus petite valeur propre de la matrice de covariance,
   orienté dans le sens moyen des lipides voisins (vers le solvant).
3. Graphe : arête i-j si |h_i - h_j| < seuil ET n_i . n_j > cos(``max_angle``)
   ET les deux têtes sont dans le même plan local (|d_ij . n| < ``max_dz``)
   ET chaque lipide est orienté comme sa normale (angle(o_i, n_i) < ``max_tilt``).
   Le seuil est cherché automatiquement comme en V1, mais sur ce graphe filtré :
   deux lipides proches mais d'orientations opposées (p.ex. cholestérol au centre,
   zones très courbées) ne sont plus reliés.
4. Les lipides restés hors des deux feuillets (cholestérol en cours de
   flip-flop, lipides isolés) sont rattachés au feuillet voisin dont la normale
   est la plus alignée avec leur propre orientation.
"""
from __future__ import annotations

import time

import numpy as np

from . import geometry, graph, v2_autoref


def orientations(s, lip, heads):
    X = geometry.make_whole(s.xyz, lip, s.box)
    sums = np.zeros((lip.n, 3))
    for k in range(3):
        sums[:, k] = np.bincount(lip.atom_lipid, weights=X[:, k], minlength=lip.n)
    centroid = sums / lip.size[:, None]
    # tête dans le même repère que le centre (image minimale)
    h = centroid + geometry.minimum_image(s.xyz[heads] - centroid, s.box)
    o = h - centroid
    o /= np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
    return h, o


def local_normals(n, i, j, d, o, same_orient=True):
    """Normales locales par ACP des déplacements vers les voisins (vectorisé)."""
    if same_orient:
        m = (o[i] * o[j]).sum(1) > 0
        i, j, d = i[m], j[m], d[m]
    # paires symétriques : j voit -d
    I = np.r_[i, j]
    D = np.r_[d, -d]
    cnt = np.bincount(I, minlength=n).astype(float) + 1.0  # +1 : le lipide lui-même (d = 0)
    mu = np.stack([np.bincount(I, D[:, k], minlength=n) for k in range(3)], 1) / cnt[:, None]
    C = np.empty((n, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            C[:, a, b] = np.bincount(I, D[:, a] * D[:, b], minlength=n) / cnt - mu[:, a] * mu[:, b]
            C[:, b, a] = C[:, a, b]
    w, v = np.linalg.eigh(C)
    nrm = v[:, :, 0]
    # orientation moyenne du voisinage pour fixer le signe
    osum = o.copy()
    for k in range(3):
        osum[:, k] += np.bincount(I, o[np.r_[j, i], k], minlength=n)
    sign = np.sign((nrm * osum).sum(1))
    sign[sign == 0] = 1
    nrm *= sign[:, None]
    # trop peu de voisins : on prend l'orientation moyenne
    few = cnt < 4
    nrm[few] = osum[few] / (np.linalg.norm(osum[few], axis=1, keepdims=True) + 1e-12)
    return nrm


def attach_leftovers(pred, i, j, d, o, nrm, dist, r_attach, flat_cos=0.5):
    """Rattache les lipides non assignés à un feuillet voisin.

    * lipide bien orienté (|o_i . n| > ``flat_cos``) : feuillet dont la normale
      moyenne est la plus alignée avec son orientation ;
    * lipide « à plat » (typiquement un cholestérol au centre de la bicouche) :
      l'orientation n'est pas informative, on prend le feuillet dont la surface
      (plan local des têtes) est la plus proche de sa tête.
    """
    left = pred < 0
    if not left.any():
        return pred
    out = pred.copy()
    m = dist < r_attach
    # D[k] = h_J - h_I
    I, J, D = np.r_[i[m], j[m]], np.r_[j[m], i[m]], np.r_[d[m], -d[m]]
    sel = left[I] & (pred[J] >= 0)
    I, J, D = I[sel], J[sel], D[sel]
    n = len(pred)
    cnt = np.zeros((n, 2))
    np.add.at(cnt, (I, pred[J]), 1.0)
    ori = np.zeros((n, 2))
    np.add.at(ori, (I, pred[J]), (o[I] * nrm[J]).sum(1))
    # profondeur de la tête i sous la surface du feuillet de j : (h_j - h_i) . n_j >= 0
    depth = np.zeros((n, 2))
    np.add.at(depth, (I, pred[J]), np.abs((D * nrm[J]).sum(1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        ori_m = ori / cnt
        depth_m = np.where(cnt > 0, depth / cnt, np.inf)
    idx = np.flatnonzero(left & (cnt.sum(1) > 0))
    a = ori_m[idx]
    oriented = np.nanmax(np.abs(a), axis=1) > flat_cos
    choice = np.where(oriented, np.nanargmax(np.nan_to_num(a, nan=-9), axis=1),
                      np.argmin(depth_m[idx], axis=1))
    out[idx] = choice
    return out


def assign(s, lip, heads=None, r_normal=2.0, max_angle=60.0, max_dz=1.0, max_tilt=60.0, cmax=3.0, min_frac=0.1,
           cutoff=None, attach=True):
    t0 = time.perf_counter()
    info = {}
    if heads is None:
        heads, _, ainfo = v2_autoref.auto_reference(s, lip)
        info["autoref_time"] = ainfo["time"]
    t1 = time.perf_counter()
    h, o = orientations(s, lip, heads)
    rmax = max(cmax, r_normal)
    i, j, dist, d = geometry.pairs_within(h, s.box, rmax, return_vectors=True)
    mn = dist < r_normal
    nrm = local_normals(lip.n, i[mn], j[mn], d[mn], o)
    ok = (nrm[i] * nrm[j]).sum(1) > np.cos(np.radians(max_angle))
    # lipides « bien formés » : orientation cohérente avec la normale locale.
    # Les lipides couchés au centre de la bicouche (cholestérol en flip-flop)
    # n'entrent pas dans le graphe ; ils sont rattachés ensuite.
    aligned = (o * nrm).sum(1) > np.cos(np.radians(max_tilt))
    ok &= aligned[i] & aligned[j]
    if max_dz is not None:
        # les deux têtes doivent être dans le même plan local (écart le long de la normale)
        ok &= (np.abs((d * nrm[i]).sum(1)) < max_dz) & (np.abs((d * nrm[j]).sum(1)) < max_dz)
    ii, jj, dd = i[ok], j[ok], dist[ok]
    if cutoff is None:
        cutoff, ginfo = graph.auto_cutoff(lip.n, ii, jj, dd, cmax, min_frac)
        info.update(ginfo)
        if cutoff is None:
            cutoff = cmax
    labels = graph.components(lip.n, ii, jj, dd < cutoff)
    pred = graph.largest_groups(labels)
    info["n_left_before_attach"] = int((pred < 0).sum())
    if attach:
        pred = attach_leftovers(pred, i, j, d, o, nrm, dist, rmax)
    info.update(cutoff=cutoff, time=time.perf_counter() - t1 + info.get("autoref_time", 0.0),
                graph_time=time.perf_counter() - t1)
    info["normals"] = nrm
    return pred, info
