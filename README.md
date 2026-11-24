# Analyse de la structure des membranes lipidiques

Projet CHPS1001 (M2 CHPS, URCA 2026/2027) : outil d'assignation des lipides d'une
membrane à leur feuillet, rapide et fiable, appliqué à des structures puis à une
trajectoire complète.

## Organisation

```
leaflets/          paquet Python
  io.py            lecture GRO rapide (numpy), lecture/écriture des index de feuillets
  topology.py      détection des lipides, sélection d'atomes de référence
  geometry.py      PBC (orthorhombique et triclinique), recherche de voisins (cKDTree)
  graph.py         composantes connexes + recherche automatique du seuil (union-find)
  evaluate.py      comparaison avec les fichiers leaflets.txt de référence
  v1_graph.py      V1 : graphe + seuil automatique
scripts/           scripts de benchmark et de comparaison
results/           résultats (tableaux, figures)
```

## Données

Les données (`Projet3.zip`) ne sont pas versionnées. Décompresser dans `data/` :

```
data/<systeme>/<systeme>.gro
data/<systeme>/leaflets.txt
```

## Installation

```
pip install -r requirements.txt
```
