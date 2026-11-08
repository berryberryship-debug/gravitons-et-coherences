# Gravitons et Cohérence : Géométrie Pré-Spatiale

Ce dépôt contient les travaux de recherche fondamentale sur l'émergence de la gravité quantique à partir de contraintes de cohérence informationnelle, inspirés par l'ordre implicite de David Bohm, le modèle holographique de Karl Pribram, et la théorie des cordes (variétés de Calabi-Yau).

## 🧠 Hypothèse Centrale
La gravité n'est pas une force fondamentale, mais une pénalité mathématique (une contrainte) visant à maintenir la cohérence entre l'ordre explicite (l'espace-temps 4D observable) et l'ordre implicite (l'information non-locale). Les lois physiques émergent comme les points fixes d'un système auto-cohérent.

## 🛠️ Formalisation Mathématique
Le modèle repose sur l'étude de matrices stochastiques de transition \(W(\alpha)\) (représentant la géométrie de l'espace) :
- **Invariant de masse** : Conservation de la ressource relationnelle (\(\Delta x_1 + \Delta x_2 = 0\)).
- **Identité spectrale** : Le spectre (valeurs propres) reste invariant sous certaines transformations, tandis que la distribution stationnaire \(\pi(\alpha)\) varie, dictant la viabilité du système.
- **Flot de Toda** : Utilisé pour déformer la géométrie de manière isospectrale. La positivité de la matrice (la "réalité" physique) est préservée si et seulement si la valeur propre minimale \(\lambda_{min} \ge 0\).

## 🔬 Résultats Numériques (Divergence Critique)
Nous avons testé la robustesse de la géométrie sous le flot de Toda sur trois chaînes de Markov (Curie-Weiss, Ehrenfest, et Ehrenfest "paresseuse"). 
Le résultat principal est la mise en évidence d'une **divergence logarithmique** du temps critique \(t_c\) (le moment où la géométrie se brise) pour la chaîne d'Ehrenfest paresseuse :

\[
 t_c \approx \frac{n}{2} \ln\left(\frac{1}{\epsilon}\right) 
\]

- **Ajustement** : \(R^2 = 0.999771\)
- **Pente mesurée** : \(a = 16.1235\) (proche de la prédiction théorique \(n/2 = 15.5\) pour \(n=31\)).

## 📂 Structure du Dépôt
- `test_phase_transition.py` : Script de simulation du flot de Toda et d'ajustement logarithmique.
- `resultats_tc_ehrenfest.csv` : Données brutes des temps critiques pour différentes valeurs d'asymétrie.
- `resultats_tc_ehrenfest.json` : Métadonnées et résultats formatés pour l'archivage.
- `Theorie.md` (à venir) : Formalisation théorique complète.

## ⚙️ Installation et Exécution
Pour reproduire les résultats :

```bash
# Installer la dépendance
pip install numpy

# Lancer la simulation
python test_phase_transition.py
