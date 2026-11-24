"""V5 : assignation des feuillets sur toutes les trames d'une trajectoire.

* topologie (GRO) lue une fois : détection des lipides + têtes automatiques (V2)
  -> calculées une seule fois et partagées par toutes les trames ;
* lecture des trames avec le lecteur XTC/TRR de MDAnalysis (code C) ;
* parallélisme « embarrassingly parallel » : la trajectoire est découpée en
  blocs contigus de trames, chaque processus ouvre son propre lecteur et
  traite son bloc (aucune communication) ;
* sorties : matrice (trames x lipides) des feuillets (int8, .npz), temps,
  nombre de changements de feuillet (flip-flops) par lipide.

Usage :
  python scripts/run_trajectory.py --top conf.gro --traj traj.xtc --nproc 4 --out results/traj
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from leaflets import read_gro  # noqa: E402
from leaflets.v4_final import LeafletAssigner  # noqa: E402

warnings.filterwarnings("ignore")
_G = {}


def box_matrix(dimensions):
    from MDAnalysis.lib.mdamath import triclinic_vectors
    return triclinic_vectors(dimensions).astype(np.float64) / 10.0  # Å -> nm


def _init(assigner, traj, n_atoms_top):
    import MDAnalysis as mda  # noqa: F401
    from MDAnalysis.coordinates import TRR, XTC
    _G["a"] = assigner
    reader = XTC.XTCReader if traj.endswith(".xtc") else TRR.TRRReader
    _G["r"] = reader(traj)
    _G["n"] = n_atoms_top


def _work(chunk):
    start, stop, stride = chunk
    a, r = _G["a"], _G["r"]
    out, times, t_read = [], [], 0.0
    t = time.perf_counter()
    for ts in r[start:stop:stride]:
        xyz = ts.positions[:_G["n"]].astype(np.float64) / 10.0
        box = box_matrix(ts.dimensions)
        t1 = time.perf_counter()
        t_read += t1 - t
        pred, info = a.assign(xyz, box)
        t = time.perf_counter()
        times.append(t - t1)
        out.append(pred.astype(np.int8))
    return start, np.array(out), np.array(times), t_read


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", required=True, help="structure GRO correspondant aux atomes de la trajectoire")
    p.add_argument("--traj", required=True, help="trajectoire XTC ou TRR")
    p.add_argument("--exclude", nargs="*", default=None, help="résidus non lipidiques à ignorer (ex. S14)")
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--begin", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--chunks-per-proc", type=int, default=4)
    p.add_argument("--out", default="results/trajectory")
    args = p.parse_args()

    t_all = time.perf_counter()
    s = read_gro(args.top)
    a = LeafletAssigner(s, exclude=args.exclude)
    t_setup = time.perf_counter() - t_all
    from MDAnalysis.coordinates import TRR, XTC
    reader = XTC.XTCReader(args.traj) if args.traj.endswith(".xtc") else TRR.TRRReader(args.traj)
    n_frames = reader.n_frames
    if reader.n_atoms < s.n_atoms:
        raise SystemExit(f"La trajectoire a {reader.n_atoms} atomes < topologie ({s.n_atoms})")
    end = n_frames if args.end is None else min(args.end, n_frames)
    frames = np.arange(args.begin, end, args.stride)
    n_chunks = max(1, args.nproc * args.chunks_per_proc)
    bounds = np.array_split(frames, n_chunks)
    chunks = [(int(b[0]), int(b[-1]) + 1, args.stride) for b in bounds if len(b)]
    print(f"{len(frames)} trames, {a.lip.n} lipides, {args.nproc} processus, {len(chunks)} blocs", flush=True)

    t_comp = time.perf_counter()
    if args.nproc == 1:
        _init(a, args.traj, s.n_atoms)
        results = [_work(c) for c in chunks]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.nproc, initializer=_init, initargs=(a, args.traj, s.n_atoms)) as pool:
            results = pool.map(_work, chunks)
    t_comp = time.perf_counter() - t_comp
    results.sort(key=lambda x: x[0])
    L = np.concatenate([r[1] for r in results])
    ft = np.concatenate([r[2] for r in results])
    t_read = sum(r[3] for r in results)

    # cohérence des étiquettes entre trames : on aligne chaque trame sur la précédente
    for f in range(1, len(L)):
        if np.mean(L[f] == L[f - 1]) < 0.5:
            L[f] = np.where(L[f] >= 0, 1 - L[f], L[f])
    changes = (L[1:] != L[:-1]).sum(0)
    total = time.perf_counter() - t_all
    summary = dict(
        topology=args.top, trajectory=args.traj, n_frames=int(len(L)), n_lipids=int(a.lip.n),
        nproc=args.nproc, setup_s=t_setup, wall_total_s=total, wall_frames_s=t_comp,
        assign_ms_per_frame_mean=float(1000 * ft.mean()), assign_ms_per_frame_median=float(1000 * np.median(ft)),
        assign_cpu_s_total=float(ft.sum()), read_cpu_s_total=float(t_read),
        throughput_frames_per_s=float(len(L) / t_comp),
        leaflet_sizes_mean=[float((L == 0).sum(1).mean()), float((L == 1).sum(1).mean())],
        n_lipids_changing_leaflet=int((changes > 0).sum()),
        head_atoms=a.head_types if hasattr(a, "head_types") else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + ".npz", leaflets=L, frame_times=ft, resindex=a.lip.resindex,
                        resname=a.lip.resname.astype("U8"))
    Path(str(out) + ".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "head_atoms"}, indent=1))


if __name__ == "__main__":
    main()
