Précision (vs leaflets.txt) / temps de calcul par structure

| Méthode | chol | peptide | prot | model_ves | dppc_ves | small_PM | large_PM |
|---|---|---|---|---|---|---|---|
| V1 noms + seuil auto | 99.74 % / 22 ms | 100.00 % / 1 ms | 100.00 % / 1 ms | 100.00 % / 36 ms | 100.00 % / 74 ms | 98.50 % / 60 ms | 98.27 % / 175 ms |
| V2 tête auto + seuil auto | 99.79 % / 21 ms | 100.00 % / 1 ms | 100.00 % / 1 ms | 100.00 % / 23 ms | 100.00 % / 68 ms | 98.36 % / 70 ms | 98.08 % / 171 ms |
| V3 sans filtre plan/inclinaison | 99.90 % / 33 ms | 100.00 % / 3 ms | 100.00 % / 2 ms | 100.00 % / 44 ms | 100.00 % / 86 ms | 99.67 % / 109 ms | 99.78 % / 345 ms |
| V3 complète (ACP, seuil auto) | 99.85 % / 38 ms | 100.00 % / 3 ms | 100.00 % / 2 ms | 100.00 % / 51 ms | 100.00 % / 88 ms | 99.71 % / 149 ms | 99.73 % / 383 ms |
| V4 ACP, seuil fixe | 99.79 % / 20 ms | 100.00 % / 3 ms | 100.00 % / 3 ms | 100.00 % / 33 ms | 100.00 % / 44 ms | 99.67 % / 76 ms | 99.72 % / 282 ms |
| V4 normale moyenne, seuil fixe | 99.79 % / 17 ms | 100.00 % / 2 ms | 100.00 % / 2 ms | 100.00 % / 26 ms | 100.00 % / 39 ms | 99.73 % / 54 ms | 99.75 % / 182 ms |
| LeafletFinder 15 Å | 99.74 % / 224 ms | 100.00 % / 4 ms | 100.00 % / 2 ms | 100.00 % / 81 ms | 100.00 % / 138 ms | 51.56 % / 362 ms | 51.47 % / 2674 ms |
| LeafletFinder seuil optimisé | 99.74 % / 5826 ms | 100.00 % / 149 ms | 100.00 % / 119 ms | 100.00 % / 6192 ms | 100.00 % / 8295 ms | 98.45 % / 22576 ms | 98.19 % / 138915 ms |
| FATSLiM | 100.00 % / 239 ms | 100.00 % / 187 ms | 100.00 % / 183 ms | plantage | 100.00 % / 219 ms | 99.82 % / 337 ms | plantage |
