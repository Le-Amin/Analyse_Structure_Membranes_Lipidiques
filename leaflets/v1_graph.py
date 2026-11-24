"""Version 1 : graphe sur un atome de référence par lipide + seuil automatique.

Sommets : un atome de référence par lipide (par défaut le phosphate, ``PO4``/``P``,
ou son équivalent pour les lipides qui n'en ont pas, cf. ``DEFAULT_REF``).
Arêtes  : deux sommets sont reliés si leur distance (PBC) est < seuil.
Feuillets : les deux plus grosses composantes connexes.
Le seuil est cherché automatiquement (voir ``graph.auto_cutoff``).
"""
from __future__ import annotations

import time

import numpy as np

from . import geometry, graph
from .topology import reference_atoms_by_name

# Ordre de priorité des noms d'atomes de référence (UA puis Martini CG)
DEFAULT_REF = ["P", "P8", "PO4", "ROH", "CP", "AM1", "GL1", "GM1"]


def assign(s, lip, ref_names=DEFAULT_REF, cmax=3.0, min_frac=0.1, cutoff=None, ref_atoms=None):
    t0 = time.perf_counter()
    if ref_atoms is None:
        ref_atoms = reference_atoms_by_name(s, lip, ref_names)
    pts = s.xyz[ref_atoms]
    i, j, dist = geometry.pairs_within(pts, s.box, cmax)
    info = {}
    if cutoff is None:
        cutoff, info = graph.auto_cutoff(lip.n, i, j, dist, cmax, min_frac)
        if cutoff is None:
            cutoff = cmax
    labels = graph.components(lip.n, i, j, dist < cutoff)
    pred = graph.largest_groups(labels)
    info.update(cutoff=cutoff, time=time.perf_counter() - t0)
    return pred, info
