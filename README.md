# Schéma de type splitting pour l’équation de Schrödinger non linéaire stochastique

### Mots-clés : Analyse numérique, Équations aux dérivées partielles, Équation de Schrödinger non linéaire, Bruit additif, Processus de Wiener, Schéma de splitting, Méthode pseudo-spectrale, Python

### Projet : Implémentation et expérimentation d’un schéma de type splitting pour l’équation de Schrödinger non linéaire stochastique.

Un schéma de type splitting est un schéma temporel qui, dans notre cas, consiste à résoudre séparément la partie linéaire et non linéaire de l’équation de Schrödinger non linéaire stochastique.
Un schéma de ce type permet d'obtenir des résultats performants, et notamment de respecter la formule de la trace pour l'évolution de la masse.
On s'intéressera à différentes non linéarités : potentiel externe, interaction non locale cubique. 
La discrétisation spatiale sera effectuée par une méthode pseudo-spectrale, mettant en œuvre la transformée de Fourier. 

Les résultats obtenus seront comparés à ceux obtenus par l’article de C.-E. Bréhier et D. Cohen donné comme référence pour ce projet (voir le rapport). De plus, les résultats obtenus par le schéma de type splitting seront comparés à ceux obtenus par d’autres schémas temporels : schéma d'Euler-Maruyama, schéma exponentiel stochastique.

Les résultats ainsi que plus de détails sont disponibles dans le rapport du projet.

Remerciements à M. Honoré pour ce sujet et son aide.
