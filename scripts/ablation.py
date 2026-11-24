"""Étude d'ablation : contribution de chaque ingrédient (précision + temps).

Usage : python scripts/ablation.py [data_dir] [n_repeats]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from leaflets import find_lipids, read_gro  # noqa: E402
from leaflets import v1_graph, v2_autoref, v3_normals, v4_final  # noqa: E402
from leaflets.evaluate import load_reference, score  # noqa: E402

SYSTEMS = ["bilayer_chol", "bilayer_peptide", "bilayer_prot", "model_vesicle",
           "dppc_vesicle", "small_plasma_membrane", "large_plasma_membrane"]


def variants(s, lip, heads):
    return {
        "V1 noms + seuil auto": lambda: v1_graph.assign(s, lip),
        "V2 tête auto + seuil auto": lambda: v1_graph.assign(s, lip, ref_atoms=heads),
        "V3 sans filtre plan/inclinaison": lambda: v3_normals.assign(s, lip, heads=heads, max_dz=None, max_tilt=180.0),
        "V3 complète (ACP, seuil auto)": lambda: v3_normals.assign(s, lip, heads=heads),
        "V4 ACP, seuil fixe": lambda: v4_final.LeafletAssigner(s, heads=heads, normal="pca").assign(s.xyz, s.box),
        "V4 normale moyenne, seuil fixe": lambda: v4_final.LeafletAssigner(s, heads=heads).assign(s.xyz, s.box),
    }


def main():
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    rep = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rows = []
    for nm in SYSTEMS:
        s = read_gro(str(data / nm / f"{nm}.gro"))
        lip = find_lipids(s)
        ref = load_reference(str(data / nm / "leaflets.txt"), lip)
        heads, _, _ = v2_autoref.auto_reference(s, lip)
        for vname, fn in variants(s, lip, heads).items():
            fn()  # échauffement (compilation numba, caches)
            ts = []
            for _ in range(rep):
                t = time.perf_counter()
                pred, info = fn()
                ts.append(time.perf_counter() - t)
            sc = score(pred, ref)
            rows.append(dict(system=nm, variant=vname, time_ms=1000 * float(np.median(ts)), **sc,
                             cutoff=float(info.get("cutoff", np.nan))))
            print(f"{nm:24s} {vname:34s} acc={sc['accuracy']*100:7.3f}%  err={sc['n_errors']:4d} "
                  f"t={rows[-1]['time_ms']:8.1f} ms  cut={rows[-1]['cutoff']:.2f}", flush=True)
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "ablation.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
