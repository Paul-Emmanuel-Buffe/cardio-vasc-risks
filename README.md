# Cardiovascular Risk Prediction System

## Description du projet
Ce projet implémente un pipeline complet (end-to-end) d'ingestion de données, d'analyse exploratoire et de modélisation prédictive pour l'évaluation des risques cardio-vasculaires. L'objectif est de concevoir un classifieur binaire capable d'émettre une probabilité de risque clinique en optimisant le compromis entre rigueur statistique et contraintes du domaine médical (minimisation des faux négatifs).

Le projet compare trois approches de modélisation : une régression logistique industrielle (Scikit-Learn), une régression logistique codée "from scratch" intégrant une régularisation L2, et un modèle non linéaire (Random Forest).

## Stack Technique
- Langage : Python 3.10+
- Base de données : PostgreSQL 18 (déploiement conteneurisé via Docker)
- Ingestion & ORM : Psycopg2, SQLAlchemy
- Data Science & ML : Pandas, NumPy, Scikit-Learn
- Visualisation : Matplotlib, Seaborn

## Structure du dépôt
- `data/` : Répertoire contenant le jeu de données brut (`cardio_train.csv`).
- `docker-compose.yml` : Configuration du conteneur PostgreSQL (mappage du port externe sur le 5433).
- `notebooks/exploration.ipynb` : Ingestion des données brutes, validation des schémas SQL, analyse descriptive, traitement des valeurs aberrantes physiologiques et étude des distributions.
- `notebooks/modelisation.ipynb` : Pipeline de feature engineering, encodage, validation croisée, entraînement, comparaison des modèles (Scikit-Learn vs Scratch vs Random Forest) et simulation d'inférence clinique.
- `requirements.txt` : Liste des dépendances Python requises.

## Architecture et Pipeline de Déploiement

### 1. Ingestion et Stockage (Cellules 1 à 6 de l'Exploration)
Le fichier brut CSV est parsé et injecté en masse (bulk insert via `execute_values`) dans une base PostgreSQL relationnelle. 
Une table nettoyée (`patients_cleaned`) est générée directement en SQL via une requête CTAS (Create Table As Select) appliquant des filtres physiologiques stricts :
- Exclusion des pressions artérielles négatives ou cliniquement impossibles.
- Cohérence hémodynamique : Pression Artérielle Systolique (PAS) obligatoirement supérieure à la Pression Artérielle Diastolique (PAD) + 10 mmHg.
- Filtrage des aberrations anthropométriques (tailles et poids physiologiquement invalides).
- Calcul natif de l'Indice de Masse Corporelle (IMC/BMI) et de son interaction avec le cholestérol.

### 2. Prétraitement et Feature Engineering (Modelisation)
- Encodage : Application d'un One-Hot Encoding (`pd.get_dummies`) avec suppression du premier témoin (`drop_first=True`) sur les variables catégorielles (`cholesterol`, `gluc`, `gender`) afin d'éviter les pièges de multicolinéarité.
- Alignement : Structuration d'une matrice de caractéristiques $X$ isolant les variables explicatives de la cible binaire $y$ (`cardio`).

### 3. Protocole d'Évaluation et Validation Croisée
Pour garantir la fiabilité scientifique et éviter toute fuite de données (*Data Leakage*), le protocole suivant est implémenté au sein d'une validation croisée stratifiée à 5 plis (`StratifiedKFold`) :
- L'ajustement d'échelle (`StandardScaler`) est instancié et ajusté (`fit_transform`) exclusivement sur le sous-ensemble d'entraînement de chaque pli, puis appliqué (`transform`) sur le sous-ensemble de test.
- Le déséquilibre des classes est traité via l'argument `class_weight='balanced'`.

### 4. Optimisation Clinique du Seuil de Décision
Dans un contexte médical, un faux négatif (patient à risque non détecté) est plus critique qu'un faux positif. Le seuil de classification par défaut (0.50) a été abaissé à **0.40** suite à l'analyse des courbes ROC AUC et Precision-Recall. Cet arbitrage permet de faire passer la sensibilité (Recall) de la classe critique à plus de 80%, tout en maintenant une précision acceptable (67.5%).

### 5. Algorithme From Scratch
La classe `LogisticRegressionScratch` implémente les fonctions mathématiques de base de la régression logistique sans bibliothèque tierce :
- Fonction d'activation : Sigmoïde avec écrêtage (`np.clip`) pour prévenir les débordements numériques (*overflow*).
- Fonction de coût : Entropie croisée binaire (Binary Cross-Entropy / Log-Loss) combinée à une pénalité de régularisation L2 (Ridge).
- Optimisation : Descente de gradient batch avec critères de convergence basés sur une tolérance d'évolution du coût ($\Delta \text{Loss} < 10^{-6}$).

## Installation et Exécution

1. Initialiser le conteneur de la base de données :
```bash
docker-compose up -d