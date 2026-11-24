"""Identification des lipides et de leurs atomes à partir d'une structure."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import Structure

WATER_IONS = {
    "W", "WF", "SOL", "WAT", "HOH", "TIP3", "TIP4", "SPC", "PW",
    "NA", "NA+", "CL", "CL-", "K", "K+", "CA", "CA2+", "MG", "ZN", "ION", "NC3+", "CLA", "SOD", "POT",
}
AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HIE", "HID", "HIP", "HSD", "HSE",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "ACE", "NH2", "NME",
    "CYX", "LYN", "ASH", "GLH",
}


@dataclass
class Lipids:
    """Description compacte (format CSR) des lipides d'un système."""
    resindex: np.ndarray   # (L,) indice séquentiel du résidu (0-based) de chaque lipide
    resname: np.ndarray    # (L,) type de lipide
    start: np.ndarray      # (L,) premier atome du lipide
    stop: np.ndarray       # (L,) dernier atome + 1
    atom_lipid: np.ndarray # (Nlip_atoms,) indice du lipide de chaque atome lipidique
    atoms: np.ndarray      # (Nlip_atoms,) indices des atomes lipidiques dans la structure

    @property
    def n(self) -> int:
        return len(self.resindex)

    @property
    def size(self) -> np.ndarray:
        return self.stop - self.start


def find_lipids(s: Structure, lipid_names=None) -> Lipids:
    """Les lipides sont les résidus qui ne sont ni eau/ions ni acides aminés
    (ou ceux dont le nom est dans ``lipid_names``) et qui ont au moins 2 atomes."""
    change = np.flatnonzero(np.r_[True, s.resindex[1:] != s.resindex[:-1]])
    starts = change
    stops = np.r_[change[1:], s.n_atoms]
    rnames = s.resname[starts]
    if lipid_names is not None:
        keep = np.isin(rnames, list(lipid_names))
    else:
        excl = np.array(sorted(WATER_IONS | AMINO_ACIDS))
        keep = ~np.isin(rnames, excl) & ((stops - starts) >= 2)
    starts, stops, rnames = starts[keep], stops[keep], rnames[keep]
    sizes = stops - starts
    atom_lipid = np.repeat(np.arange(len(starts)), sizes)
    offsets = np.r_[0, np.cumsum(sizes)[:-1]]
    atoms = starts[atom_lipid] + (np.arange(sizes.sum()) - offsets[atom_lipid])
    return Lipids(s.resindex[starts], rnames, starts, stops, atom_lipid, atoms)


def reference_atoms_by_name(s: Structure, lip: Lipids, priority) -> np.ndarray:
    """Pour chaque lipide, renvoie l'indice du premier atome dont le nom
    apparaît dans ``priority`` (par ordre de priorité). À défaut : premier atome."""
    names = s.name[lip.atoms]
    rank = np.full(len(names), len(priority), dtype=np.int64)
    for k, nm in enumerate(priority[::-1]):
        rank[names == nm] = len(priority) - 1 - k
    # tri par (lipide, rang, position) puis premier élément de chaque lipide
    order = np.lexsort((np.arange(len(names)), rank, lip.atom_lipid))
    first = np.r_[True, lip.atom_lipid[order][1:] != lip.atom_lipid[order][:-1]]
    return lip.atoms[order[first]]
