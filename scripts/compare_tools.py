"""Comparaison avec LeafletFinder (MDAnalysis) et FATSLiM sur les 7 structures.

Les deux outils reçoivent les mêmes atomes de tête que notre V1 (PO4/P/ROH…).
* LeafletFinder : seuil par défaut 15 Å, puis seuil optimisé (``optimize_cutoff``)
  — le temps de l'optimisation est compté.
* FATSLiM : ``fatslim membranes`` (seuil 2 nm par défaut, normales locales),
  exécuté dans son propre environnement (Python 3.8, chemin via $FATSLIM).

Usage : FATSLIM=/chemin/vers/fatslim python scripts/compare_tools.py [data]
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from leaflets import find_lipids, read_gro, write_leaflet_file  # noqa: E402
from leaflets.evaluate import load_reference, score  # noqa: E402
from leaflets.io import read_leaflet_file  # noqa: E402
from leaflets.topology import reference_atoms_by_name  # noqa: E402
from leaflets.v1_graph import DEFAULT_REF  # noqa: E402

warnings.filterwarnings("ignore")
SYSTEMS = ["bilayer_chol", "bilayer_peptide", "bilayer_prot", "model_vesicle",
           "dppc_vesicle", "small_plasma_membrane", "large_plasma_membrane"]


def groups_to_pred(groups_resindex, lip):
    """groups_resindex : liste de tableaux d'indices séquentiels de résidus (0-based)."""
    pos = {r: k for k, r in enumerate(lip.resindex)}
    pred = np.full(lip.n, -1, dtype=np.int64)
    sizes = sorted(range(len(groups_resindex)), key=lambda g: -len(groups_resindex[g]))
    for rank, g in enumerate(sizes[:2]):
        for r in groups_resindex[g]:
            k = pos.get(int(r))
            if k is not None:
                pred[k] = rank
    return pred


def run_leafletfinder(gro, heads, lip, s, optimize):
    import MDAnalysis as mda
    from MDAnalysis.analysis.leaflet import LeafletFinder, optimize_cutoff
    u = mda.Universe(gro)
    sel = u.atoms[heads]
    t = time.perf_counter()
    cutoff = 15.0
    if optimize:
        try:
            cutoff, _ = optimize_cutoff(u, sel, dmin=10.0, dmax=30.0, step=0.5, pbc=True)
        except IndexError:
            # optimize_cutoff ne trouve aucun seuil donnant 2 groupes « équilibrés »
            # (cas des vésicules : feuillets interne/externe très inégaux) -> 15 Å
            cutoff = 15.0
    lf = LeafletFinder(u, sel, cutoff=cutoff, pbc=True)
    groups = [g.residues.resindices for g in lf.groups()]
    dt = time.perf_counter() - t
    return groups_to_pred(groups, lip), dt, cutoff / 10.0


def run_fatslim(gro, heads, lip, s, exe, nthreads=4):
    with tempfile.TemporaryDirectory() as tmp:
        ndx = os.path.join(tmp, "hg.ndx")
        write_leaflet_file(ndx, {"headgroups": heads + 1})
        out = os.path.join(tmp, "out.ndx")
        t = time.perf_counter()
        res = subprocess.run([exe, "membranes", "-c", gro, "-n", ndx, "--output-index", out,
                              "--nthreads", str(nthreads)], capture_output=True, text=True)
        wall = time.perf_counter() - t
        m = re.search(r"done in ([\d.]+) ms", res.stdout)
        frame_ms = float(m.group(1)) / 1000 if m else np.nan
        files = sorted(Path(tmp).glob("out_*.ndx"))
        if not files:
            return np.full(lip.n, -1), wall, frame_ms
        g = read_leaflet_file(str(files[0]))
        # FATSLiM écrit les indices (1-based) de tous les atomes des lipides de chaque feuillet
        groups = [np.unique(s.resindex[ids - 1]) for ids in g.values()]
        return groups_to_pred(groups, lip), wall, frame_ms


def main():
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    exe = os.environ.get("FATSLIM", "fatslim")
    rows = []
    for nm in SYSTEMS:
        gro = str(data / nm / f"{nm}.gro")
        s = read_gro(gro)
        lip = find_lipids(s)
        ref = load_reference(str(data / nm / "leaflets.txt"), lip)
        heads = reference_atoms_by_name(s, lip, DEFAULT_REF)
        for name, fn in [
            ("LeafletFinder 15 Å", lambda: run_leafletfinder(gro, heads, lip, s, False)),
            ("LeafletFinder seuil optimisé", lambda: run_leafletfinder(gro, heads, lip, s, True)),
            ("FATSLiM", lambda: run_fatslim(gro, heads, lip, s, exe)),
        ]:
            try:
                pred, dt, extra = fn()
            except Exception as e:  # noqa: BLE001
                print(nm, name, "ECHEC", e)
                rows.append(dict(system=nm, tool=name, error=str(e)))
                continue
            sc = score(pred, ref)
            row = dict(system=nm, tool=name, time_ms=1000 * dt, **sc)
            if name == "FATSLiM":
                row["frame_ms"] = 1000 * extra
            else:
                row["cutoff"] = extra
            rows.append(row)
            print(f"{nm:24s} {name:30s} acc={sc['accuracy']*100:7.3f}% err={sc['n_errors']:5d} "
                  f"t={row['time_ms']:9.1f} ms", flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/compare_tools.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
