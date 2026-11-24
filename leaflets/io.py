"""Lecture / écriture des fichiers GRO et des fichiers d'index de feuillets.

Le lecteur GRO est écrit à la main (sans MDAnalysis) pour être rapide :
on lit le fichier en bloc et on découpe les colonnes à largeur fixe avec numpy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Structure:
    title: str
    resid: np.ndarray      # numéro de résidu tel qu'écrit dans le fichier (modulo 100000)
    resindex: np.ndarray   # indice séquentiel du résidu (0-based), robuste au repliement des numéros
    resname: np.ndarray    # nom de résidu (str)
    name: np.ndarray       # nom d'atome (str)
    xyz: np.ndarray        # (N, 3) float64, en nm
    box: np.ndarray        # (3, 3) vecteurs de boîte en lignes, en nm

    @property
    def n_atoms(self) -> int:
        return len(self.name)


def box_from_gro(values) -> np.ndarray:
    v = [float(x) for x in values]
    box = np.zeros((3, 3))
    box[0, 0], box[1, 1], box[2, 2] = v[0], v[1], v[2]
    if len(v) == 9:
        # ordre GRO : v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
        box[0, 1], box[0, 2] = v[3], v[4]
        box[1, 0], box[1, 2] = v[5], v[6]
        box[2, 0], box[2, 1] = v[7], v[8]
    return box


def read_gro(path: str) -> Structure:
    with open(path) as fh:
        lines = fh.read().split("\n")
    title = lines[0]
    n = int(lines[1].strip())
    atoms = lines[2:2 + n]
    box = box_from_gro(lines[2 + n].split())

    # La précision des coordonnées est variable : la largeur des champs
    # est donnée par la distance entre deux points décimaux.
    first = atoms[0]
    dots = [m.start() for m in re.finditer(r"\.", first[20:])]
    width = dots[1] - dots[0]

    arr = np.array(atoms, dtype=f"U{max(len(a) for a in atoms)}")
    b = arr.view("U1").reshape(len(arr), -1)

    def col(a, z):
        return np.char.strip(b[:, a:z].copy().view(f"U{z - a}").ravel())

    resid = col(0, 5).astype(np.int64)
    resname = col(5, 10)
    name = col(10, 15)
    xyz = np.empty((n, 3))
    for k in range(3):
        s, e = 20 + k * width, 20 + (k + 1) * width
        xyz[:, k] = col(s, e).astype(np.float64)

    # indice séquentiel de résidu : nouveau résidu dès que (resid, resname) change
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (resid[1:] != resid[:-1]) | (resname[1:] != resname[:-1])
    resindex = np.cumsum(change) - 1
    return Structure(title, resid, resindex, resname, name, xyz, box)


def read_leaflet_file(path: str) -> dict[str, np.ndarray]:
    """Lit un fichier type .ndx (``[ groupe ]`` puis des entiers)."""
    groups: dict[str, list[int]] = {}
    current = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("["):
                current = line.strip("[] ")
                groups[current] = []
            elif current is not None:
                groups[current].extend(int(x) for x in line.split())
    return {k: np.array(v, dtype=np.int64) for k, v in groups.items()}


def write_leaflet_file(path: str, groups: dict[str, np.ndarray], per_line: int = 15) -> None:
    with open(path, "w") as fh:
        for gname, ids in groups.items():
            fh.write(f"[ {gname} ]\n")
            ids = np.asarray(ids)
            for i in range(0, len(ids), per_line):
                fh.write("".join(f"{x:6d}" for x in ids[i:i + per_line]) + "\n")
