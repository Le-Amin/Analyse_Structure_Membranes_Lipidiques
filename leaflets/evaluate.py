"""Comparaison d'une assignation de feuillets avec la référence fournie."""
from __future__ import annotations

import numpy as np

from .io import read_leaflet_file


def load_reference(path, lip):
    """Renvoie un tableau (L,) avec 0/1 pour les lipides de la référence, -1 sinon.

    Dans les fichiers fournis, les indices sont les numéros séquentiels (1-based)
    des résidus dans le fichier GRO.
    """
    groups = read_leaflet_file(path)
    ref = np.full(lip.n, -1, dtype=np.int64)
    pos = {r: k for k, r in enumerate(lip.resindex + 1)}
    for g, (_, ids) in enumerate(groups.items()):
        for r in ids:
            k = pos.get(int(r))
            if k is not None:
                ref[k] = g
    return ref


def score(pred, ref):
    """Précision après la meilleure permutation des étiquettes (0<->1).

    Les lipides non assignés (-1) comptent comme des erreurs.
    """
    m = ref >= 0
    p, r = pred[m], ref[m]
    acc_direct = np.mean(p == r)
    swapped = np.where(p >= 0, 1 - p, -1)
    acc_swap = np.mean(swapped == r)
    acc = max(acc_direct, acc_swap)
    best = p if acc_direct >= acc_swap else swapped
    return {
        "accuracy": float(acc),
        "n_ref": int(m.sum()),
        "n_errors": int((best != r).sum()),
        "n_unassigned": int((p < 0).sum()),
    }
