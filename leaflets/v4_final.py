"""Version 4 (finale) : uniquement les éléments nécessaires et suffisants.

Choix retenus (justifiés par l'étude d'ablation, ``scripts/ablation.py``) :

* **Atome de tête automatique (V2)**, calculé *une seule fois par topologie* et
  mis en cache : générique (65 types de lipides, UA ou CG), aucun nom à fournir.
* **Orientation + normale locale (V3)** : c'est ce qui élimine les ponts entre
  feuillets créés par les cholestérols du centre de la bicouche et rend le
  résultat insensible au seuil. La normale est l'orientation moyenne des
  voisins (``normal="mean"``, défaut) : même précision que l'ACP mais moins de
  calcul ; l'ACP reste disponible (``normal="pca"``).
* **Seuil fixe = rayon de voisinage** : une fois le graphe filtré par les
  normales, les feuillets ne fusionnent plus quel que soit le seuil (≤ 3 nm),
  la recherche automatique devient inutile à chaque trame. Elle est conservée
  en option (``cutoff=None``) et utilisée en sécurité si le graphe ne donne
  pas deux grosses composantes.
* **Une seule recherche de voisins** par trame (cKDTree périodique), réutilisée
  pour la normale, le graphe et le rattachement des lipides isolés.
"""
from __future__ import annotations

import time

import numpy as np

from . import geometry, graph, v2_autoref
from .topology import find_lipids
from .v3_normals import attach_leftovers, local_normals


def mean_normals(n, i, j, o):
    """Normale locale = moyenne des orientations des voisins de même sens."""
    m = (o[i] * o[j]).sum(1) > 0
    i, j = i[m], j[m]
    acc = o.copy()
    for k in range(3):
        acc[:, k] += np.bincount(i, o[j, k], minlength=n) + np.bincount(j, o[i, k], minlength=n)
    return acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-12)


class LeafletAssigner:
    """Assignation des feuillets, optimisée pour une trajectoire.

    La partie « topologie » (lipides, têtes) est calculée une fois à la création ;
    ``assign(xyz, box)`` traite ensuite une trame.
    """

    def __init__(self, structure, lipid_names=None, exclude=None, heads=None, radius=2.0, max_angle=60.0,
                 max_dz=1.0, max_tilt=60.0, normal="mean", cutoff="radius", min_frac=0.1):
        t0 = time.perf_counter()
        self.lip = find_lipids(structure, lipid_names, exclude)
        if heads is None:
            heads, _, info = v2_autoref.auto_reference(structure, self.lip)
            self.head_types = info["per_type"]
        self.heads = heads
        self.radius = radius
        self.cos_angle = np.cos(np.radians(max_angle))
        self.cos_tilt = np.cos(np.radians(max_tilt))
        self.max_dz = max_dz
        self.normal = normal
        self.cutoff = radius if cutoff == "radius" else cutoff
        self.min_frac = min_frac
        lip = self.lip
        self._inv_size = 1.0 / lip.size
        self.setup_time = time.perf_counter() - t0

    # ------------------------------------------------------------------
    def _orient(self, xyz, box):
        lip = self.lip
        X = geometry.make_whole(xyz, lip, box)
        c = np.empty((lip.n, 3))
        for k in range(3):
            c[:, k] = np.bincount(lip.atom_lipid, weights=X[:, k], minlength=lip.n)
        c *= self._inv_size[:, None]
        o = geometry.minimum_image(xyz[self.heads] - c, box)
        h = c + o
        o /= np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
        return h, o

    def assign(self, xyz, box):
        """Renvoie (pred, info) ; pred[k] ∈ {0, 1} est le feuillet du lipide k."""
        t0 = time.perf_counter()
        n = self.lip.n
        h, o = self._orient(xyz, box)
        rmax = self.radius if self.cutoff is None else max(self.radius, self.cutoff)
        i, j, dist, d = geometry.pairs_within(h, box, rmax, return_vectors=True)
        if self.normal == "pca":
            m = dist < self.radius
            nrm = local_normals(n, i[m], j[m], d[m], o)
        else:
            m = dist < self.radius
            nrm = mean_normals(n, i[m], j[m], o)
        aligned = (o * nrm).sum(1) > self.cos_tilt
        ok = aligned[i] & aligned[j] & ((nrm[i] * nrm[j]).sum(1) > self.cos_angle)
        if self.max_dz is not None:
            ok &= (np.abs((d * nrm[i]).sum(1)) < self.max_dz) & (np.abs((d * nrm[j]).sum(1)) < self.max_dz)
        ii, jj, dd = i[ok], j[ok], dist[ok]
        cutoff = self.cutoff
        info = {}
        if cutoff is None:
            cutoff, info = graph.auto_cutoff(n, ii, jj, dd, rmax, self.min_frac)
            cutoff = rmax if cutoff is None else cutoff
        labels = graph.components(n, ii, jj, dd < cutoff)
        pred = graph.largest_groups(labels)
        # sécurité : la 2e composante doit être une vraie couche
        if (pred == 1).sum() < self.min_frac * n and self.cutoff is not None:
            c2, _ = graph.auto_cutoff(n, ii, jj, dd, rmax, self.min_frac)
            if c2 is not None:
                cutoff = c2
                pred = graph.largest_groups(graph.components(n, ii, jj, dd < cutoff))
        info["n_left_before_attach"] = int((pred < 0).sum())
        pred = attach_leftovers(pred, i, j, d, o, nrm, dist, rmax)
        # convention : feuillet 0 = celui vers lequel pointent les normales
        # (supérieur pour une bicouche plane, externe pour une vésicule)
        pred = self._canonical(pred, h, nrm, box)
        info.update(cutoff=cutoff, time=time.perf_counter() - t0)
        return pred, info

    @staticmethod
    def _canonical(pred, h, nrm, box):
        """Feuillet 0 = feuillet supérieur (bicouche plane) ou externe (vésicule)."""
        a, b = pred == 0, pred == 1
        if not a.any() or not b.any():
            return pred
        na, nb = nrm[a].mean(0), nrm[b].mean(0)
        if np.linalg.norm(na) > 0.5 and np.linalg.norm(nb) > 0.5:
            flip = na[2] < nb[2]          # bicouche plane : normale vers +z => supérieur
        else:                             # vésicule : normales radiales, moyenne ~ 0
            ctr = h.mean(0)
            sa = ((h[a] - ctr) * nrm[a]).sum(1).mean()
            sb = ((h[b] - ctr) * nrm[b]).sum(1).mean()
            flip = sa < sb                # externe : normale sortante
        if flip:
            pred = np.where(pred >= 0, 1 - pred, pred)
        return pred


def assign(s, lip=None, **kw):
    """Interface identique aux autres versions (une structure)."""
    t0 = time.perf_counter()
    a = LeafletAssigner(s, **kw)
    pred, info = a.assign(s.xyz, s.box)
    info["setup_time"] = a.setup_time
    info["frame_time"] = info["time"]
    info["time"] = time.perf_counter() - t0
    return pred, info
