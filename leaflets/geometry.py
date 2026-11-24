"""Géométrie avec conditions périodiques (PBC) et recherche de voisins.

* Boîte orthorhombique : ``scipy.spatial.cKDTree`` avec ``boxsize`` (PBC natif).
* Boîte triclinique : on passe en coordonnées fractionnaires, on replie les
  points dans la boîte, puis on ajoute des images périodiques des points proches
  des bords (« halo ») avant de construire l'arbre.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def is_orthorhombic(box: np.ndarray) -> bool:
    return np.allclose(box - np.diag(np.diag(box)), 0.0)


def minimum_image(d: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Applique la convention de l'image minimale à des vecteurs de déplacement."""
    if is_orthorhombic(box):
        L = np.diag(box)
        return d - L * np.round(d / L)
    inv = np.linalg.inv(box)
    f = d @ inv
    f -= np.round(f)
    return f @ box


def make_whole(xyz: np.ndarray, lip, box: np.ndarray) -> np.ndarray:
    """Reconstruit les lipides coupés par les bords de boîte (référence = 1er atome)."""
    x = xyz[lip.atoms]
    anchor = xyz[lip.start][lip.atom_lipid]
    return anchor + minimum_image(x - anchor, box)


def wrap(xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    if is_orthorhombic(box):
        L = np.diag(box)
        return xyz - L * np.floor(xyz / L)
    inv = np.linalg.inv(box)
    f = xyz @ inv
    f -= np.floor(f)
    return f @ box


class PeriodicTree:
    """cKDTree périodique pour une boîte quelconque.

    ``idx[k]`` donne l'indice original du k-ième point de l'arbre (les images
    périodiques ajoutées dans le halo pointent vers le point d'origine).
    """

    def __init__(self, points, box, cutoff):
        self.box = box
        n = len(points)
        p = wrap(points, box)
        if is_orthorhombic(box):
            L = np.diag(box)
            p = np.minimum(p, np.nextafter(L, 0))
            self.tree = cKDTree(p, boxsize=L)
            self.idx = np.arange(n)
            self.halo = False
        else:
            f = p @ np.linalg.inv(box)
            heights = np.abs(np.linalg.det(box)) / np.linalg.norm(
                np.cross(box[[1, 2, 0]], box[[2, 0, 1]]), axis=1)
            margin = cutoff / heights
            xs, ids = [p], [np.arange(n)]
            for sh in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3):
                if not sh.any():
                    continue
                g = f + sh
                m = np.all((g > -margin) & (g < 1 + margin), axis=1)
                if m.any():
                    xs.append(g[m] @ box)
                    ids.append(np.flatnonzero(m))
            self.tree = cKDTree(np.concatenate(xs))
            self.idx = np.concatenate(ids)
            self.halo = True
        self.points = p

    def query_points(self, q):
        """Points de requête ramenés dans le repère de l'arbre."""
        q = wrap(q, self.box)
        if not self.halo:
            q = np.minimum(q, np.nextafter(np.diag(self.box), 0))
        return q


def pairs_within(points: np.ndarray, box: np.ndarray, cutoff: float, return_vectors=False):
    """Toutes les paires (i < j) à distance < cutoff, avec PBC.

    Renvoie (i, j, dist) et optionnellement les vecteurs d_ij = x_j - x_i (image minimale).
    """
    t = PeriodicTree(points, box, cutoff)
    pr = t.tree.query_pairs(cutoff, output_type="ndarray")
    if t.halo:
        pr = t.idx[pr]
        pr = pr[pr[:, 0] != pr[:, 1]]
        pr.sort(axis=1)
        pr = np.unique(pr, axis=0)
    i, j = pr[:, 0], pr[:, 1]
    d = minimum_image(points[j] - points[i], box)
    dist = np.sqrt((d * d).sum(1))
    if return_vectors:
        return i, j, dist, d
    return i, j, dist
