# PLAN EXERCICE 2: Feature Selection Hybride (AG + SA)

## 1. Objectif Global
Résoudre le problème de **Feature Selection** en utilisant une approche hybride combinant:
- **Algorithme Génétique (AG)** pour une recherche globale et rapide.
- **Simulated Annealing (SA)** pour un affinement local de la meilleure solution trouvée par l'AG.

Le but est de maximiser la performance de classification (Accuracy) tout en minimisant le nombre d'attributs sélectionnés (Réduction de la dimensionalité).

## 2. Architecture des Fichiers
L'architecture du projet est modulaire:
- `1_data_loading.py` : Import et préparation des datasets (Wine, Zoo, Krvskp).
- `2_fitness_function.py` : Calcul du score selon la formule (0.7 * Accuracy - 0.3 * Reduction).
- `3_genetic_algorithm.py` : Opérateurs génétiques (Sélection, Croisement, Mutation).
- `4_simulated_annealing.py` : Critère de Metropolis et refroidissement.
- `5_hybrid_algorithm.py` : Orchestration de l'hybridation (AG -> SA).
- `config.py` : Hyperparamètres des algorithmes.
- `7_main_experiment.py` : Exécution du pipeline complet sur les données.
- `8_results_analysis.py` : Visualisations et métriques (Graphiques de convergence).

## 3. Flux d'exécution
1. **Initialisation** : Chargement des paramètres depuis `config.py`.
2. **Préparation des données** : Chargement, séparation (train/test) et normalisation.
3. **Phase 1 (Recherche Globale)** : Lancement de l'AG sur 50 générations. L'AG retourne la meilleure combinaison binaire de features.
4. **Phase 2 (Affinement Local)** : Le SA prend le résultat de l'AG comme point de départ et l'optimise selon un schéma de température décroissante.
5. **Évaluation** : Entraînement d'un classifieur final (ex: SVM) avec uniquement les features sélectionnées par le SA.
6. **Rapports** : Génération de tableaux CSV et de graphiques PNG dans le dossier `results/`.

## 4. Résultats Attendus
Pour chaque dataset (Wine, Zoo, Krvskp), nous devrions observer:
- Une diminution drastique du nombre de features (ex: passer de 13 à 5 ou 6).
- Le maintien ou l'amélioration de la précision (Accuracy).
- Une courbe de convergence dans le dossier `results` démontrant que l'hybridation trouve de meilleures solutions qu'un seul algorithme.
