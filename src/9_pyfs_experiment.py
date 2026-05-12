import sys
import os
import pandas as pd
import numpy as np

# S'assurer de pouvoir importer nos propres modules
sys.path.append('.')
import importlib.util

def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

DataLoader = import_module_from_file("data_loading", "1_data_loading.py").DataLoader
FitnessFunction = import_module_from_file("fitness_function", "2_fitness_function.py").FitnessFunction

# --- PATCH POUR Py_FS ---
# Py_FS essaie d'importer 'pkg_resources' qui a été supprimé des versions récentes de Python (setuptools>=70).
# Comme nous n'utilisons pas les datasets internes de Py_FS, nous pouvons "simuler" ce module pour éviter le crash.
import sys
import types
if 'pkg_resources' not in sys.modules:
    sys.modules['pkg_resources'] = types.ModuleType('pkg_resources')

import builtins
def smart_input(prompt=""):
    print(f"[Py_FS demande] {prompt}")
    p = prompt.lower()
    if 'weight' in p: return '0.7'
    if 'val' in p or 'size' in p: return '0.3'
    return '1'
builtins.input = smart_input

import sklearn.model_selection
original_split = sklearn.model_selection.train_test_split
def safe_split(*args, **kwargs):
    # Forcer des paramètres sûrs pour éviter le crash interne de Py_FS
    if 'stratify' in kwargs: del kwargs['stratify']
    kwargs['test_size'] = 0.3
    return original_split(*args, **kwargs)
sklearn.model_selection.train_test_split = safe_split
# ------------------------

# Import de la librairie Py_FS
from Py_FS.wrapper.nature_inspired import PSO, GWO, WOA

def run_pyfs_experiments():
    loader = DataLoader()
    datasets = loader.load_all_datasets('../datasets')
    
    if 'wine' not in datasets:
        print("Dataset Wine introuvable dans ../datasets/. Veuillez le rajouter.")
        return
        
    data = datasets['wine']
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    n_features = data['n_features']
    
    # Nous utilisons la MEME fonction d'évaluation que l'Ex 2 pour une comparaison juste (SVM)
    fitness_evaluator = FitnessFunction(X_train, X_test, y_train, y_test)
    
    results = []
    
    # 1. PSO (Particle Swarm Optimization)
    print("\n[1/3] Exécution Py_FS : Particle Swarm Optimization (PSO)...")
    sol_pso = PSO(num_agents=30, max_iter=50, train_data=X_train, train_label=y_train, save_conv_graph=False)
    acc_pso, red_pso = fitness_evaluator.get_accuracy_and_reduction(sol_pso.best_agent)
    fit_pso = fitness_evaluator.evaluate_solution(sol_pso.best_agent)
    results.append({
        'Méthode': 'Py_FS : PSO',
        'Fitness': round(fit_pso, 4),
        'Accuracy': f"{acc_pso*100:.2f}%",
        'Features Sélectionnées': f"{int(np.sum(sol_pso.best_agent))} / {n_features}",
        'Réduction': f"{red_pso:.2f}%"
    })
    
    # 2. GWO (Grey Wolf Optimizer)
    print("\n[2/3] Exécution Py_FS : Grey Wolf Optimizer (GWO)...")
    sol_gwo = GWO(num_agents=30, max_iter=50, train_data=X_train, train_label=y_train, save_conv_graph=False)
    acc_gwo, red_gwo = fitness_evaluator.get_accuracy_and_reduction(sol_gwo.best_agent)
    fit_gwo = fitness_evaluator.evaluate_solution(sol_gwo.best_agent)
    results.append({
        'Méthode': 'Py_FS : GWO',
        'Fitness': round(fit_gwo, 4),
        'Accuracy': f"{acc_gwo*100:.2f}%",
        'Features Sélectionnées': f"{int(np.sum(sol_gwo.best_agent))} / {n_features}",
        'Réduction': f"{red_gwo:.2f}%"
    })
    
    # 3. WOA (Whale Optimization Algorithm)
    print("\n[3/3] Exécution Py_FS : Whale Optimization Algorithm (WOA)...")
    sol_woa = WOA(num_agents=30, max_iter=50, train_data=X_train, train_label=y_train, save_conv_graph=False)
    acc_woa, red_woa = fitness_evaluator.get_accuracy_and_reduction(sol_woa.best_agent)
    fit_woa = fitness_evaluator.evaluate_solution(sol_woa.best_agent)
    results.append({
        'Méthode': 'Py_FS : WOA',
        'Fitness': round(fit_woa, 4),
        'Accuracy': f"{acc_woa*100:.2f}%",
        'Features Sélectionnées': f"{int(np.sum(sol_woa.best_agent))} / {n_features}",
        'Réduction': f"{red_woa:.2f}%"
    })
    
    # 4. Ajout de notre algorithme Hybride de l'Exercice 2
    results.append({
        'Méthode': 'AG + SA (Notre Hybride Ex2)',
        'Fitness': 0.6178,
        'Accuracy': '98.15%',
        'Features Sélectionnées': '3 / 13',
        'Réduction': '76.92%'
    })
    
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("      TABLEAU COMPARATIF FINAL (WINE DATASET)")
    print("="*50)
    print(df.to_string(index=False))
    
    csv_path = '../results/comparaison_ex2_ex3.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Tableau sauvegardé dans: {csv_path}")
    
    # Création et sauvegarde automatique du graphique comparatif
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    # Couleurs personnalisées (style viridis)
    colors = ['#440154', '#3b528b', '#21918c', '#fde725']
    bars = plt.bar(df['Méthode'], df['Fitness'], color=colors)
    
    plt.title('Comparaison des Algorithmes (Fitness) - WINE Dataset', fontsize=14, pad=15)
    plt.ylabel('Score Fitness', fontsize=12)
    plt.xlabel('Algorithmes', fontsize=12)
    plt.ylim(0, 0.8)
    
    # Ajouter les valeurs au-dessus des barres
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    fig_path = '../results/comparaison_ex3_fitness.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✅ Graphique comparatif sauvegardé dans: {fig_path}")

if __name__ == "__main__":
    run_pyfs_experiments()
