"""Étape 6 sur la trajectoire : LeafletFinder et FATSLiM contre notre V4.

Pour chaque outil : temps total et accord trame par trame avec la V4
(fraction de lipides assignés au même feuillet, après la meilleure permutation).

Usage :
  FATSLIM=/chemin/fatslim python scripts/compare_trajectory.py --top prod.gro --traj traj.xtc \
      --v4 results/trajectory.npz --exclude S14 [--lf-frames 200]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from leaflets import find_lipids, read_gro, write_leaflet_file  # noqa: E402
from leaflets.io import read_leaflet_file  # noqa: E402
from leaflets.topology import reference_atoms_by_name  # noqa: E402
from leaflets.v1_graph import DEFAULT_REF  # noqa: E402

warnings.filterwarnings("ignore")


def agreement(a, b):
    m = (a >= 0) & (b >= 0)
    same = np.mean(a == b)
    swap = np.mean(np.where(a >= 0, 1 - a, -1) == b)
    return float(max(same, swap)), float(np.mean(~m))


def to_pred(groups_atoms, atom2lip, n):
    """groups_atoms : indices d'atomes (0-based) de chaque groupe -> feuillet par lipide."""
    pred = np.full(n, -1, dtype=np.int8)
    groups = [np.unique(atom2lip[g][atom2lip[g] >= 0]) for g in groups_atoms]
    order = sorted(range(len(groups)), key=lambda g: -len(groups[g]))
    for rank, g in enumerate(order[:2]):
        pred[groups[g]] = rank
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", required=True)
    p.add_argument("--traj", required=True)
    p.add_argument("--v4", required=True, help="sortie .npz de run_trajectory.py")
    p.add_argument("--exclude", nargs="*", default=None)
    p.add_argument("--lf-frames", type=int, default=None, help="nb de trames pour LeafletFinder (lent)")
    p.add_argument("--nthreads", type=int, default=4)
    p.add_argument("--out", default="results/compare_trajectory.json")
    args = p.parse_args()

    s = read_gro(args.top)
    lip = find_lipids(s, exclude=args.exclude)
    heads = reference_atoms_by_name(s, lip, DEFAULT_REF)
    atom2lip = np.full(s.n_atoms, -1)
    atom2lip[lip.atoms] = lip.atom_lipid
    V4 = np.load(args.v4)["leaflets"]
    nf = len(V4)
    res = {}

    # ---------------- FATSLiM (lit lui-même la trajectoire) ----------------
    exe = os.environ.get("FATSLIM", "fatslim")
    with tempfile.TemporaryDirectory() as tmp:
        ndx = os.path.join(tmp, "hg.ndx")
        write_leaflet_file(ndx, {"headgroups": heads + 1})
        out = os.path.join(tmp, "out.ndx")
        t = time.perf_counter()
        r = subprocess.run([exe, "membranes", "-c", args.top, "-t", args.traj, "-n", ndx,
                            "--output-index", out, "--nthreads", str(args.nthreads)],
                           capture_output=True, text=True)
        wall = time.perf_counter() - t
        # FATSLiM nomme les fichiers par le temps (out_<t>.ndx) : tri numérique
        files = sorted(Path(tmp).glob("out_*.ndx"), key=lambda f: float(f.stem.split("_")[-1]))
        acc = []
        for f, fn in enumerate(files[:nf]):
            g = read_leaflet_file(str(fn))
            acc.append(agreement(to_pred([ids - 1 for ids in g.values()], atom2lip, lip.n), V4[f])[0])
        res["FATSLiM"] = dict(wall_s=wall, n_frames=len(files), ms_per_frame=1000 * wall / max(1, len(files)),
                              agreement_with_V4_mean=float(np.mean(acc)) if acc else None,
                              agreement_with_V4_min=float(np.min(acc)) if acc else None,
                              stderr_tail=r.stderr[-300:] if r.returncode else "")
        print("FATSLiM", res["FATSLiM"], flush=True)

    # ---------------- LeafletFinder (MDAnalysis) ----------------
    import MDAnalysis as mda
    from MDAnalysis.analysis.leaflet import LeafletFinder, optimize_cutoff
    u = mda.Universe(args.top, args.traj)
    sel = u.atoms[heads]
    n_lf = nf if args.lf_frames is None else min(nf, args.lf_frames)
    t = time.perf_counter()
    try:
        cutoff, _ = optimize_cutoff(u, sel, dmin=10.0, dmax=30.0, step=0.5, pbc=True)
    except IndexError:  # aucun seuil dans [10, 30] Å ne donne 2 groupes équilibrés
        cutoff = None
    t_opt = time.perf_counter() - t
    runs = [("LeafletFinder 15 Å", 15.0)]
    if cutoff is not None:
        runs.append((f"LeafletFinder seuil optimisé ({cutoff:.1f} Å)", cutoff))
    res["LeafletFinder optimize_cutoff"] = dict(time_s=t_opt, cutoff_A=cutoff)
    for label, c in runs:
        acc, t = [], time.perf_counter()
        for f, ts in enumerate(u.trajectory[:n_lf]):
            lf = LeafletFinder(u, sel, cutoff=c, pbc=True)
            acc.append(agreement(to_pred([g.indices for g in lf.groups()], atom2lip, lip.n), V4[f])[0])
        wall = time.perf_counter() - t
        res[label] = dict(wall_s=wall, n_frames=n_lf, ms_per_frame=1000 * wall / n_lf,
                          extrapolated_all_frames_s=wall / n_lf * nf,
                          optimize_cutoff_s=t_opt if c != 15.0 else 0.0,
                          agreement_with_V4_mean=float(np.mean(acc)), agreement_with_V4_min=float(np.min(acc)))
        print(label, res[label], flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
