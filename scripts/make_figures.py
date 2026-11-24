"""Tableaux (Markdown) et figures à partir des JSON de résultats."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
SYS = ["bilayer_chol", "bilayer_peptide", "bilayer_prot", "model_vesicle",
       "dppc_vesicle", "small_plasma_membrane", "large_plasma_membrane"]
SHORT = ["chol", "peptide", "prot", "model_ves", "dppc_ves", "small_PM", "large_PM"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

abl = json.loads((R / "ablation.json").read_text())
cmp_ = json.loads((R / "compare_tools.json").read_text())
rows = [dict(method=r["variant"], **r) for r in abl] + \
       [dict(method=r["tool"], **r) for r in cmp_ if "accuracy" in r]
methods = list(dict.fromkeys(r["method"] for r in rows))
keep = ["V1 noms + seuil auto", "V2 tête auto + seuil auto", "V3 complète (ACP, seuil auto)",
        "V4 normale moyenne, seuil fixe", "LeafletFinder 15 Å", "LeafletFinder seuil optimisé", "FATSLiM"]


def get(m, s, k):
    for r in rows:
        if r["method"] == m and r["system"] == s:
            if r.get("n_unassigned", 0) == r.get("n_ref", -1):
                return None  # l'outil a planté / n'a rien renvoyé
            return r.get(k)
    return None


# ---- tableau markdown ----
lines = ["| Méthode | " + " | ".join(SHORT) + " |", "|---|" + "---|" * len(SHORT)]
for m in methods:
    cells = []
    for s in SYS:
        a, t = get(m, s, "accuracy"), get(m, s, "time_ms")
        cells.append("plantage" if a is None else f"{100*a:.2f} % / {t:.0f} ms")
    lines.append(f"| {m} | " + " | ".join(cells) + " |")
(R / "tableau_structures.md").write_text(
    "Précision (vs leaflets.txt) / temps de calcul par structure\n\n" + "\n".join(lines) + "\n")

# ---- figures ----
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b"})
x = np.arange(len(SYS))
w = 0.8 / len(keep)
for key, fname, ylabel, logy in [("accuracy", "fig_precision.png", "Précision (%)", False),
                                ("time_ms", "fig_temps.png", "Temps par structure (ms, échelle log)", True)]:
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    for k, m in enumerate(keep):
        vals = [get(m, s, key) for s in SYS]
        v = np.array([np.nan if a is None else (100 * a if key == "accuracy" else a) for a in vals])
        xs = x + (k - len(keep) / 2 + 0.5) * w
        ax.bar(xs, v, w * 0.9, label=m, color=PALETTE[k], edgecolor="#fcfcfb", linewidth=0.5)
        for xi, vi in zip(xs, v):
            if np.isnan(vi):
                ax.text(xi, 0.02, "plantage", transform=ax.get_xaxis_transform(), rotation=90,
                        ha="center", va="bottom", fontsize=6.5, color="#52514e")
    ax.set_xticks(x, SHORT)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 102)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(R / fname)
    plt.close(fig)
print((R / "tableau_structures.md").read_text())
