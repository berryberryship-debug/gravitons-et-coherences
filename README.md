# Gravitons et Cohérence : Géométrie Pré-Spatiale

Ce dépôt contient les travaux de recherche fondamentale sur l'émergence de la gravité quantique à partir de contraintes de cohérence informationnelle, inspirés par l'ordre implicite de David Bohm, le modèle holographique de Karl Pribram, et la théorie des cordes (variétés de Calabi-Yau).

## 🧠 Hypothèse Centrale
La gravité n'est pas une force fondamentale, mais une pénalité mathématique (une contrainte) visant à maintenir la cohérence entre l'ordre explicite (l'espace-temps 4D observable) et l'ordre implicite (l'information non-locale). Les lois physiques émergent comme les points fixes d'un système auto-cohérent.

## 🛠️ Formalisation Mathématique
Le modèle repose sur l'étude de matrices stochastiques de transition \(W(\alpha)\) (représentant la géométrie de l'espace) :
- **Invariant de masse** : Conservation de la ressource relationnelle (\(\Delta x_1 + \Delta x_2 = 0\)).
- **Identité spectrale** : Le spectre reste invariant sous certaines transformations, tandis que la distribution stationnaire \(\pi(\alpha)\) varie, dictant la viabilité.
- **Flot de Toda** : Utilisé pour déformer la géométrie de manière isospectrale. La positivité de la matrice est préservée si et seulement si la valeur propre minimale \(\lambda_{min} \ge 0\).

## 🔬 Découvertes Majeures

### 1. Divergence Critique (Chaîne d'Ehrenfest)
Mise en évidence d'une divergence logarithmique du temps critique \(t_c\) :
\[
 t_c \approx \frac{n}{2} \ln\left(\frac{1}{\epsilon}\right) 
\]
- **Ajustement** : \(R^2 = 0.999771\)
- **Pente mesurée** : \(a = 16.1235\) (proche de la théorie \(n/2 = 15.5\)).

### 2. Effondrement Dynamique et Spectral Flow
Mise en évidence d'un effondrement du gap spectral (évitement de croisement) sous perturbation de spin-2 (graviton), validant la présence d'un Mass Gap topologique.

### 3. 🏆 Preuve Finale : Singularité Topologique (Point Diabolique)
Le transport adiabatique du vecteur d'état le long d'un lacet fermé dans l'espace des paramètres \((A, t)\) a été simulé avec succès. 

- **Phase de Berry accumulée** : \(3.1416\) radians, soit exactement **\(\pi\)**.
- **Signature** : Inversion du vecteur propre (monodromie) à la fermeture de la boucle.

Cette phase non-triviale (invariant \(\mathbb{Z}_2\)) prouve l'existence d'un **véritable point diabolique** (cône de Dirac). La géométrie pré-spatiale possède une charge topologique robuste, immunisée contre le bruit (barrière néguentropique).

## 📂 Structure du Dépôt
- `test_phase_transition.py` : Simulation du flot de Toda et divergence critique.
- `phase_berry_finale.py` : Preuve topologique de la singularité (Phase de Berry).
- `resultats_*.csv/json` : Données brutes des simulations.
- `Theorie_Trou_Noir.md` : Formalisation mathématique détaillée.

## ⚙️ Installation et Exécution

```bash
# Installer les dépendances
pip install numpy scipy matplotlib

# Lancer la simulation principale
python test_phase_transition.py

# Lancer la preuve topologique (Phase de Berry)
python phase_berry_finale.py
