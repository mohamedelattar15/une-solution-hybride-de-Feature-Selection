# 📑 INDEX COMPLET - EXERCICE 2 FEATURE SELECTION

## 🚀 DÉMARRAGE RAPIDE (3 étapes)

1. **Lire d'abord**: [`README.md`](README.md) (10 min)
2. **Installation**: 
   ```bash
   pip install -r requirements.txt
   mkdir datasets
   # Ajouter wine.data, zoo.data, krvskp.data
   ```
3. **Lancer**: 
   ```bash
   python test_quick.py          # Test (2 min)
   python 7_main_experiment.py   # Exécution (30-60 min)
   ```

---

## 📚 DOCUMENTATION

### Pour comprendre le projet
- **[README.md](README.md)** - Vue d'ensemble, installation, utilisation
- **[GUIDE_COMPLET.md](GUIDE_COMPLET.md)** - Explication détaillée (A à Z)
- **[PLAN_EXERCICE_2.md](PLAN_EXERCICE_2.md)** - Plan du projet structuré
- **[ARBORESCENCE.md](ARBORESCENCE.md)** - Structure fichiers + flux données

### Pour configurer
- **[config.py](config.py)** - Paramètres AG, SA, Fitness
- **[requirements.txt](requirements.txt)** - Dépendances Python

---

## 🐍 CODE PYTHON (8 MODULES)

### 1️⃣ **[1_data_loading.py](1_data_loading.py)** (6.6 KB)
Charge et prépare les 3 datasets

```python
# Usage
loader = DataLoader()
datasets = loader.load_all_datasets('./datasets')
# Retourne: {'wine': {...}, 'zoo': {...}, 'krvskp': {...}}
```

**Classes**: `DataLoader`  
**Fonctions clés**: `load_wine_data()`, `prepare_data()`

---

### 2️⃣ **[2_fitness_function.py](2_fitness_function.py)** (6.9 KB)
Fonction d'évaluation des solutions

```python
# Formule de fitness
Fitness = 0.7 * Accuracy - 0.3 * (n_selected / n_total)
```

**Classes**: `FitnessFunction`, `MultiClassifierFitness`  
**Fonctions clés**: `evaluate_solution()`, `get_accuracy_and_reduction()`

---

### 3️⃣ **[3_genetic_algorithm.py](3_genetic_algorithm.py)** (9.0 KB)
Algorithme Génétique pour exploration globale

```python
ga = GeneticAlgorithm(fitness, n_features=13)
best_solution, best_fitness = ga.evolve(n_generations=50)
```

**Classes**: `GeneticAlgorithm`  
**Opérateurs**: Tournament Selection, One-point Crossover, Bit-flip Mutation

---

### 4️⃣ **[4_simulated_annealing.py](4_simulated_annealing.py)** (8.3 KB)
Simulated Annealing pour affinement local

```python
sa = SimulatedAnnealing(fitness, n_features=13)
best_solution_sa, best_fitness_sa = sa.optimize(initial_solution)
```

**Classes**: `SimulatedAnnealing`  
**Critère**: Metropolis acceptance avec température décroissante

---

### 5️⃣ **[5_hybrid_algorithm.py](5_hybrid_algorithm.py)** (11 KB)
Hybridation AG + SA

```python
hybrid = HybridGASA(ga, sa)
best_solution, best_fitness = hybrid.run(n_generations_ga=50)
```

**Classes**: `HybridGASA`  
**Phases**: AG (exploration) → SA (refinement)

---

### 6️⃣ **[config.py](config.py)** (6.7 KB)
Configuration centrale

```python
GA_CONFIG = {...}          # Paramètres AG
SA_CONFIG = {...}          # Paramètres SA
FITNESS_CONFIG = {...}     # Paramètres fitness
```

**Utile**: `get_config_profile(profile_name)`

---

### 7️⃣ **[7_main_experiment.py](7_main_experiment.py)** (10 KB) ⭐ EXÉCUTION
Script principal d'expérimentation

```python
# Lance le test complet
python 7_main_experiment.py
```

**Classes**: `FeatureSelectionExperiment`  
**Sorties**: CSV + TXT + graphiques

---

### 8️⃣ **[8_results_analysis.py](8_results_analysis.py)** (9.9 KB)
Visualisation et analyse des résultats

```python
analyzer = ResultsAnalyzer()
analyzer.generate_full_report(results_dict)
```

**Classes**: `ResultsAnalyzer`  
**Graphiques**: Convergence, Comparaison, Features

---

### 🧪 **[test_quick.py](test_quick.py)** (5.6 KB)
Test de validation rapide

```bash
python test_quick.py    # ✅ Vérifie tout fonctionne
```

**Teste**: Dépendances, loading, fitness, AG, SA, hybridation

---

## 📊 RÉSULTATS GÉNÉRÉS

Après exécution, ces fichiers sont créés dans `results/`:

```
results/
├── results_summary.csv               # Tableau comparatif CSV
├── results_comparison.png            # Bar charts
├── features_importance.png           # Features sélectionnées
│
├── wine_convergence.png              # Graphiques Wine
├── wine_detailed_results.txt
│
├── zoo_convergence.png               # Graphiques Zoo
├── zoo_detailed_results.txt
│
├── krvskp_convergence.png            # Graphiques Krvskp
└── krvskp_detailed_results.txt
```

---

## 🎯 FLUX DE TRAVAIL COMPLET

```
1. SETUP
   ├─ pip install -r requirements.txt
   ├─ mkdir datasets
   └─ Télécharger 3 datasets

2. TEST
   ├─ python test_quick.py
   └─ ✅ Vérifier succès

3. EXÉCUTION
   ├─ python 7_main_experiment.py
   ├─ Attendre 30-60 min
   └─ Résultats dans results/

4. ANALYSE
   ├─ Ouvrir results_summary.csv
   ├─ Visualiser PNG
   └─ Lire TXT détaillés
```

---

## 🔧 CONFIGURATION RAPIDE

### Profils prédéfinis
```python
from config import get_config_profile

# Quick (2-5 min)
config = get_config_profile('quick')

# Medium (10-20 min, défaut)
config = get_config_profile('medium')

# Thorough (30-60 min)
config = get_config_profile('thorough')
```

### Ajuster les paramètres
Éditer `config.py`:
```python
GA_CONFIG['population_size'] = 50      # De 30 à 50
GA_CONFIG['n_generations'] = 100       # De 50 à 100
SA_CONFIG['initial_temperature'] = 2.0 # De 1.0 à 2.0
```

---

## 📈 INTERPRÉTATION DES RÉSULTATS

### Tableau CSV
| Colonne | Signification |
|---------|---------------|
| Dataset | Wine / Zoo / Krvskp |
| Fitness | Score combiné (higher is better) |
| Accuracy | Précision avec features sélectionnées |
| Features | Nombre sélectionné / Total |
| Réduction | Pourcentage de reduction dimensionnelle |

### Graphiques
- **convergence.png** : AG converge vite, SA affine graduellement
- **comparison.png** : Comparer les 3 datasets
- **features_importance.png** : Quelles features sélectionnées

### Fichiers TXT
- **detailed_results.txt** : Rapport complet par dataset
- Inclut: Fitness, Accuracy, Features, Temps, Évaluations

---

## 💡 CONCEPTS CLÉS

### Algorithme Génétique
- **Population** : 30 chromosomes binaires
- **Sélection** : Tournament (k=3)
- **Croisement** : One-point
- **Mutation** : Bit-flip (10% par bit)
- **Générations** : 50

### Simulated Annealing
- **Voisinage** : Flip 1 bit aléatoire
- **Acceptation** : Metropolis (P = exp(ΔE/T))
- **Température** : T(t+1) = 0.95 * T(t)
- **Arrêt** : T < 0.01

### Fitness
- **Objectif 1** : Maximize Accuracy (poids 0.7)
- **Objectif 2** : Minimize Features (poids 0.3)
- **Équilibre** : 70% performance, 30% simplicité

### Hybridation
- **Phase 1** : AG explore largement
- **Phase 2** : SA affine la meilleure solution
- **Bénéfice** : Meilleur que séparément

---

## 📝 CHECKLIST COMPLÈTE

- [ ] Tous les fichiers Python (.py)
- [ ] Documentation (README, GUIDE, etc.)
- [ ] Configuration (config.py)
- [ ] Test (test_quick.py)
- [ ] Dépendances (requirements.txt)
- [ ] Datasets (wine, zoo, krvskp)
- [ ] AG implémenté correctement
- [ ] SA implémenté correctement
- [ ] Hybridation fonctionne
- [ ] Résultats cohérents
- [ ] Graphiques générés
- [ ] Rapport sauvegardé
- [ ] Code commenté
- [ ] Pas d'erreurs

---

## 🐛 TROUBLESHOOTING

### ImportError
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Données manquantes
```bash
# Télécharger manuellement dans datasets/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/krvskp.data
```

### Lent
- Réduire `population_size` ou `n_generations` dans config.py
- Utiliser profil 'quick'

### Faibles résultats
- Augmenter `population_size` et `n_generations`
- Augmenter `initial_temperature`

---

## 📞 CONTACTS RAPIDES

### Besoin d'aide sur...

| Sujet | Fichier |
|-------|---------|
| Comment exécuter | README.md |
| Expliquer en détail | GUIDE_COMPLET.md |
| Configurer paramètres | config.py |
| Structures fichiers | ARBORESCENCE.md |
| Problèmes | test_quick.py |
| Résultats | 8_results_analysis.py |

---

## 🎓 APPRENTISSAGE

### Concepts couverts
✅ Algorithmes d'optimisation  
✅ Feature Selection  
✅ Architecture modulaire  
✅ Machine Learning  
✅ Visualisation données  

### Skills développées
✅ Python avancé  
✅ OOP  
✅ Algorithmes  
✅ Data Science  
✅ Optimisation  

---

## 📊 TAILLE DU PROJET

| Élément | Taille |
|---------|--------|
| Code Python | ~100 KB |
| Documentation | ~50 KB |
| Configuration | ~7 KB |
| **Total** | **~157 KB** |

### Ligne de code
- **Code fonctionnel** : ~1200 lignes
- **Docstrings** : ~400 lignes
- **Commentaires** : ~300 lignes
- **Tests** : ~200 lignes
- **Total** : ~2100 lignes

---

## ✨ POINTS FORTS

✅ **Complet** : 8 modules + documentation + test  
✅ **Professionnel** : OOP, structure, gestion erreurs  
✅ **Documenté** : README, guides, docstrings  
✅ **Testable** : test_quick.py fourni  
✅ **Configurable** : Paramètres centralisés  
✅ **Scalable** : Facile d'étendre  
✅ **Reproductible** : random_state=42  

---

## 🚀 COMMANDES ESSENTIELLES

```bash
# Installation
pip install -r requirements.txt

# Test rapide
python test_quick.py

# Exécution complète
python 7_main_experiment.py

# Voir les résultats
ls -lh results/
cat results/results_summary.csv
```

---

**Toute la documentation est dans ce dossier. Commencez par README.md!** 📖

Dernière mise à jour: Janvier 2025  
Statut: ✅ Production Ready
