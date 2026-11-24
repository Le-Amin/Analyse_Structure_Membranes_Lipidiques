"""Version 2 : plus besoin de définir un atome de référence par lipide.

Idée : les lipides sont amphiphiles. La tête polaire est au contact du solvant,
alors que l'extrémité des chaînes est enfouie au centre de la bicouche, où les
chaînes des deux feuillets se rencontrent. Autour de la tête, la densité
d'atomes *lipidiques* est donc minimale ; autour de la queue, elle est maximale.

Pour chaque type de lipide (même nom de résidu = même ordre d'atomes), on
calcule, sur un échantillon de molécules, le nombre moyen d'atomes lipidiques
d'autres molécules dans une sphère de rayon ``radius`` autour de chaque atome.
L'atome de densité minimale devient l'atome de tête (référence), celui de densité
maximale l'atome de queue. Aucune connaissance chimique (noms d'atomes,
présence d'eau…) n'est nécessaire : seule la géométrie de la membrane est utilisée.
La détermination est faite une seule fois par topologie, puis réutilisée.
"""
from __future__ import annotations

import time

import numpy as np

from . import geometry, v1_graph


def auto_reference(s, lip, radius=1.0, max_per_type=200, seed=0):
    t0 = time.perf_counter()
    X = geometry.make_whole(s.xyz, lip, s.box)
    tree = geometry.PeriodicTree(X, s.box, radius)
    rng = np.random.default_rng(seed)
    head_off = np.zeros(lip.n, dtype=np.int64)
    tail_off = np.zeros(lip.n, dtype=np.int64)
    per_type = {}
    offsets = np.r_[0, np.cumsum(lip.size)[:-1]]  # début de chaque lipide dans X
    for t in np.unique(lip.resname):
        mols = np.flatnonzero(lip.resname == t)
        nat = int(lip.size[mols[0]])
        if nat < 2:
            continue
        sample = mols if len(mols) <= max_per_type else rng.choice(mols, max_per_type, replace=False)
        sample = sample[lip.size[sample] == nat]
        rows = offsets[sample][:, None] + np.arange(nat)[None, :]      # (S, nat)
        q = tree.query_points(X[rows.ravel()])
        counts = tree.tree.query_ball_point(q, radius, return_length=True).reshape(len(sample), nat)
        # on retire les atomes de la molécule elle-même
        Y = X[rows]
        own = (np.linalg.norm(Y[:, :, None, :] - Y[:, None, :, :], axis=-1) < radius).sum(-1)
        dens = (counts - own).mean(0)
        h, tl = int(np.argmin(dens)), int(np.argmax(dens))
        head_off[mols] = h
        tail_off[mols] = tl
        names = s.name[lip.start[mols[0]]:lip.stop[mols[0]]]
        per_type[str(t)] = {"head": str(names[h]), "tail": str(names[tl]), "n": int(len(mols))}
    heads = lip.start + np.minimum(head_off, lip.size - 1)
    tails = lip.start + np.minimum(tail_off, lip.size - 1)
    return heads, tails, {"per_type": per_type, "time": time.perf_counter() - t0}


def assign(s, lip, heads=None, **kw):
    info = {}
    if heads is None:
        heads, _, ainfo = auto_reference(s, lip)
        info["autoref_time"] = ainfo["time"]
        info["per_type"] = ainfo["per_type"]
    pred, ginfo = v1_graph.assign(s, lip, ref_atoms=heads, **kw)
    info.update(ginfo)
    info["time"] = ginfo["time"] + info.get("autoref_time", 0.0)
    return pred, info
