# GUIDE COMPLET - EXERCICE 2 FEATURE SELECTION

## 🎯 OBJECTIF FINAL

**Implémenter une solution hybride de Feature Selection combinant:**
- ✅ Algorithme Génétique (AG) pour l'exploration globale
- ✅ Simulated Annealing (SA) pour l'affinement local
- ✅ Tester sur 3 datasets: Wine, Zoo, Krvskp

---

## 📂 STRUCTURE DU CODE (A à Z)

### MODULE 1: Chargement des données (`1_data_loading.py`)
**Responsabilité**: Charger, nettoyer et préparer les données

**Classes principales**:
- `DataLoader` : Charge les 3 datasets
  - `load_wine_data()` : Dataset Wine (178 samples, 13 features, 3 classes)
  - `load_zoo_data()` : Dataset Zoo
  - `load_krvskp_data()` : Dataset Chess
  - `prepare_data()` : Normalise et split 70/30

**Sortie**: 
```python
datasets = {
    'wine': {
        'X_train': array(120, 13),
        'X_test': array(58, 13),
        'y_train': array(120,),
        'y_test': array(58,),
        'features': array(['Alcohol', 'Malic acid', ...]),
        'n_features': 13
    },
    ...
}
```

---

### MODULE 2: Fonction de fitness (`2_fitness_function.py`)
**Responsabilité**: Évaluer la qualité d'une solution

**Formule**:
```
Fitness = α * Accuracy - β * (n_features_selected / n_features_total)
        = 0.7 * Accuracy - 0.3 * (n_selected / n_total)
```

**Interprétation**:
- Maximize Accuracy : Performance de classification
- Minimize features : Réduction dimensionnelle

**Exemple**:
```python
chromosome = [1, 0, 1, 1, ..., 0]  # 6 features sélectionnées
accuracy = 0.95
fitness = 0.7 * 0.95 - 0.3 * (6/13) = 0.527
```

**Classe `FitnessFunction`**:
- `__init__()` : Initialise avec données et classifieur
- `evaluate_solution(chromosome)` : Retourne le score
- `get_accuracy_and_reduction(chromosome)` : Détails
- `reset_evaluation_count()` : Réinitialise le compteur

---

### MODULE 3: Algorithme Génétique (`3_genetic_algorithm.py`)
**Responsabilité**: Exploration globale de l'espace de solutions

**Principe**:
1. Population initiale aléatoire
2. Évaluer chaque chromosome
3. Sélectionner les meilleurs (tournoi)
4. Croiser (one-point crossover)
5. Muter (bit-flip)
6. Répéter N générations

**Opérateurs génétiques**:
```
Sélection: Tournament(k=3) - Sélectionner le meilleur parmi 3 aléatoires
Croisement: One-point - Couper et combiner 2 parents
Mutation: Bit-flip - Flip chaque bit avec probabilité p
Élitisme: Garder le meilleur chromosome
```

**Classe `GeneticAlgorithm`**:
- `initialize_population()` : Crée population (0,1) aléatoire
- `evaluate_population()` : Fitness pour tous
- `tournament_selection()` : Sélection par tournoi
- `crossover()` : Croisement à 1 point
- `mutation()` : Flip bits aléatoires
- `evolve(n_generations)` : Boucle principale

**Résultat**:
```python
best_solution, best_fitness = ga.evolve(50)
# best_solution: [1, 0, 1, ..., 1]  (meilleur chromosome)
# best_fitness: 0.527
```

---

### MODULE 4: Simulated Annealing (`4_simulated_annealing.py`)
**Responsabilité**: Affinement local avec acceptation probabiliste

**Principe**:
1. Partir d'une solution
2. Générer un voisin (flip 1 bit)
3. Accepter si meilleur OU avec probabilité exp(-ΔE/T)
4. Réduire température T
5. Répéter jusqu'à convergence

**Critère d'acceptation (Metropolis)**:
```
Si fitness(neighbor) > fitness(current):
    Accepter toujours
Sinon:
    Accepter avec probabilité exp(ΔE/T)
    où ΔE = fitness(neighbor) - fitness(current)
```

**Refroidissement**:
```
T(t+1) = T(t) * cooling_rate
T0 = 1.0
cooling_rate = 0.95
Arrêt quand T < 0.01
```

**Classe `SimulatedAnnealing`**:
- `get_neighbor()` : Flip 1 bit aléatoire
- `acceptance_probability()` : Calcule P(accept)
- `optimize(initial_solution)` : Boucle principale

**Résultat**:
```python
best_solution_sa, best_fitness_sa = sa.optimize(best_solution_ag)
# Généralement meilleur que AG seul!
```

---

### MODULE 5: Hybridation (`5_hybrid_algorithm.py`)
**Responsabilité**: Combiner AG + SA intelligemment

**Processus en 2 phases**:

**PHASE 1: Algorithme Génétique (0-50 générations)**
```
Objectif: Exploration globale
- Diversité: Population de 30 chromosomes
- Couverture: Croisement et mutation
- Résultat: Bonnes solutions dispersées
- Temps: ~50% du temps total
```

**PHASE 2: Simulated Annealing (Refroidissement complet)**
```
Objectif: Affinement local
- Point de départ: Meilleure solution de l'AG
- Stratégie: Accepter temporairement des solutions pires
- Convergence: Progressive avec T décroissante
- Temps: ~50% du temps total
```

**Classe `HybridGASA`**:
- `run(n_generations_ga)` : Exécute AG puis SA
- `_print_final_summary()` : Affiche les résultats
- `get_results_dict()` : Retourne dictionnaire résultats

**Bénéfices**:
```
AG seul:           Fitness = 0.500 (exploration, moins précis)
SA seul:           Fitness = 0.480 (refinement, bloqué localement)
AG + SA (hybride): Fitness = 0.527 (global + local, meilleur!)
```

---

### MODULE 6: Script principal (`7_main_experiment.py`)
**Responsabilité**: Orchestrer toute l'expérience

**Classe `FeatureSelectionExperiment`**:
- `run_dataset()` : Exécute sur 1 dataset
- `run_all_datasets()` : Lance pour Wine, Zoo, Krvskp
- `generate_report()` : Crée tableau comparatif
- `save_detailed_results()` : Sauvegarde fichiers

**Flux principal**:
```python
1. Charger 3 datasets (data_loading)
2. Pour chaque dataset:
   a. Créer FitnessFunction
   b. Créer GeneticAlgorithm
   c. Créer SimulatedAnnealing
   d. Exécuter HybridGASA
   e. Collecter résultats
3. Générer rapport comparatif
4. Sauvegarder en CSV/TXT
```

**Résultat**:
```
results/
├── results_summary.csv          # Tableau comparatif
├── wine_detailed_results.txt    # Détails Wine
├── zoo_detailed_results.txt     # Détails Zoo
└── krvskp_detailed_results.txt  # Détails Krvskp
```

---

### MODULE 7: Analyse (`8_results_analysis.py`)
**Responsabilité**: Visualiser et analyser les résultats

**Classe `ResultsAnalyzer`**:
- `plot_convergence_comparison()` : AG vs SA convergence
- `plot_results_comparison()` : Compare 3 datasets
- `plot_feature_importance()` : Quelles features sélectionnées
- `generate_full_report()` : Tous les graphiques

**Graphiques générés**:
```
1. convergence_wine.png    : Fitness AG/SA + Température + Acceptance
2. convergence_zoo.png     : Idem pour Zoo
3. convergence_krvskp.png  : Idem pour Krvskp
4. results_comparison.png  : Bar charts comparatifs
5. features_importance.png : Quelles features par dataset
```

---

## 🚀 GUIDE D'EXÉCUTION

### Étape 1: Installation
```bash
# Installer les dépendances
pip install -r requirements.txt

# Ou manuellement
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Étape 2: Télécharger les datasets
```bash
# Créer le dossier
mkdir datasets

# Télécharger les 3 fichiers:
# - datasets/wine.data
# - datasets/zoo.data
# - datasets/krvskp.data
```

### Étape 3: Test rapide
```bash
# Vérifier que tout fonctionne
python test_quick.py

# Résultat attendu: ✅ TOUS LES TESTS RÉUSSIS!
```

### Étape 4: Exécution complète
```bash
# Lancer l'expérience
python 7_main_experiment.py

# Durée: ~30-60 min (selon capacité machine)
# Résultats sauvegardés dans ./results/
```

### Étape 5: Analyse
```python
# Les graphiques sont générés automatiquement
# Fichiers PNG dans ./results/
```

---

## 📊 PARAMÈTRES CLÉS

### Algorithme Génétique
```python
population_size = 30        # Taille population
mutation_rate = 0.1        # 10% mutation par individu
crossover_rate = 0.8       # 80% chance de croisement
tournament_size = 3        # Sélection parmi 3 aléatoires
n_generations = 50         # 50 générations
```

### Simulated Annealing
```python
initial_temperature = 1.0   # T initiale
cooling_rate = 0.95        # T(t+1) = 0.95 * T(t)
iterations_per_temperature = 200  # Itérations par étape T
final_temperature = 0.01   # Condition d'arrêt
```

### Fonction de fitness
```python
alpha = 0.7                # Poids accuracy
beta = 0.3                 # Poids réduction
```

---

## 📈 RÉSULTATS ATTENDUS

### Tableau récapitulatif
```
Dataset  │ Fitness │ Accuracy │ Features │ Réduction │ Temps
─────────┼─────────┼──────────┼──────────┼───────────┼──────
Wine     │ 0.6530  │ 0.9630   │  7/13    │   46.2%   │ 12.5s
Zoo      │ 0.6240  │ 0.9200   │  8/16    │   50.0%   │ 14.2s
Krvskp   │ 0.5890  │ 0.9850   │ 18/36    │   50.0%   │ 28.3s
```

### Graphiques
- Convergence AG : Fitness augmente et se stabilise
- Convergence SA : Fitness continue à s'améliorer légèrement
- Température SA : Décroissance exponentielle
- Acceptation SA : Diminue avec température
- Comparaison datasets : Visualiser les différences

---

## 🔧 OPTIMISATION

### Si résultats faibles:
```python
# Augmenter exploration
population_size = 50        # De 30 à 50
n_generations = 100         # De 50 à 100
mutation_rate = 0.15        # De 0.1 à 0.15
initial_temperature = 2.0   # De 1.0 à 2.0
```

### Si exécution lente:
```python
# Réduire calculs
population_size = 15        # De 30 à 15
n_generations = 25          # De 50 à 25
iterations_per_temperature = 100  # De 200 à 100
```

### Si instabilité:
```python
# Plus stable
crossover_rate = 0.9        # Augmenter croisement
tournament_size = 5         # Augmenter pression sélective
cooling_rate = 0.98         # Ralentir refroidissement
```

---

## 📝 POINTS CLÉS À COMPRENDRE

### 1. **Pourquoi hybride?**
- AG : Bon pour exploration globale, mais peut converger prématurément
- SA : Bon pour refinement local, mais peut rester bloqué
- Hybride : Meilleur des deux mondes!

### 2. **Codage binaire**
- Chaque bit = une feature (1=sélectionnée, 0=non)
- Exemple : [1, 0, 1, 1, 0] = features {0, 2, 3} sélectionnées

### 3. **Fitness multi-objective**
- Balancer accuracy vs réduction
- Alpha (0.7) > Beta (0.3) : Privilégier accuracy
- Peut être ajusté selon objectif

### 4. **Metropolis acceptance**
- Accepter meilleurs : Convergence
- Accepter pires (avec proba) : Echapper optima locaux
- Probabilité baisse avec T : Progressive convergence

### 5. **Population vs voisinage**
- AG opère sur population : Diversité
- SA opère sur voisinage : Intensité locale
- Combinaison : Diversité + Intensité = Optimal!

---

## ✅ CHECKLIST AVANT SOUMISSION

- [ ] Code bien structuré en 8 modules
- [ ] Tous les fichiers .py commentés
- [ ] README.md complet
- [ ] PLAN_EXERCICE_2.md détaillé
- [ ] requirements.txt correct
- [ ] test_quick.py passe sans erreur
- [ ] 3 datasets chargés correctement
- [ ] Algorithme génétique implémenté correctement
- [ ] Simulated Annealing implémenté correctement
- [ ] Hybridation AG+SA fonctionnelle
- [ ] Résultats sauvegardés (CSV + graphiques)
- [ ] Rapport final généré
- [ ] Pas d'erreurs à l'exécution
- [ ] Performance raisonnable (~1-2 min par dataset)

---

## 🎓 CONCEPTS APPRIS

✅ **Algorithmes d'optimisation**
- Exploration vs exploitation
- Population-based vs trajectory-based
- Hybridation de méthodes

✅ **Feature Selection**
- Représentation binaire
- Fitness multi-objective
- Trade-off accuracy-complexité

✅ **Programmation**
- Architecture modulaire
- OOP (Classes)
- Gestion de données

✅ **Machine Learning**
- Classifieurs (SVM, DT, NB)
- Validation croisée
- Performance metrics

---

**FIN DU GUIDE COMPLET**

Pour toute question, consultez:
1. README.md
2. Docstrings dans les modules
3. Fichiers de configuration (config.py)
4. Script de test (test_quick.py)

Bonne chance! 🚀
